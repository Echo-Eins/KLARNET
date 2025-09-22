// crates/nlu/src/patterns.rs

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

use crate::NluProcessor;
use klarnet_core::{
    CommandType, Entity, Intent, KlarnetError, KlarnetResult, LocalCommand, NluResult,
};

pub struct PatternMatcher {
    patterns: Vec<IntentPattern>,
    entities: HashMap<String, EntityExtractor>,
}

#[derive(Debug, Clone)]
struct IntentPattern {
    name: String,
    patterns: Vec<Regex>,
    confidence: f32,
    action: Option<String>,
}

#[derive(Debug, Clone)]
struct EntityExtractor {
    name: String,
    pattern: Regex,
    entity_type: EntityType,
}

#[derive(Debug, Clone)]
enum EntityType {
    Room,
    Device,
    Number,
    Color,
    Time,
    Custom(String),
}

impl PatternMatcher {
    pub async fn new(patterns_file: Option<String>) -> KlarnetResult<Self> {
        let patterns = Self::load_patterns(patterns_file)?;
        let entities = Self::create_entity_extractors();

        Ok(Self { patterns, entities })
    }

    fn load_patterns(_patterns_file: Option<String>) -> KlarnetResult<Vec<IntentPattern>> {
        // In production, load from YAML file
        // For now, hardcode common patterns

        Ok(vec![
            IntentPattern {
                name: "lights_on".to_string(),
                patterns: vec![
                    Regex::new(r"включи(ть)?\s+свет").unwrap(),
                    Regex::new(r"зажги(ть)?\s+свет").unwrap(),
                    Regex::new(r"свет\s+включи(ть)?").unwrap(),
                ],
                confidence: 0.9,
                action: Some("smart_home.lights".to_string()),
            },
            IntentPattern {
                name: "lights_off".to_string(),
                patterns: vec![
                    Regex::new(r"выключи(ть)?\s+свет").unwrap(),
                    Regex::new(r"погаси(ть)?\s+свет").unwrap(),
                    Regex::new(r"свет\s+выключи(ть)?").unwrap(),
                ],
                confidence: 0.9,
                action: Some("smart_home.lights".to_string()),
            },
            IntentPattern {
                name: "open_app".to_string(),
                patterns: vec![
                    Regex::new(r"открой(ть)?\s+(\w+)").unwrap(),
                    Regex::new(r"запусти(ть)?\s+(\w+)").unwrap(),
                    Regex::new(r"включи(ть)?\s+(\w+)").unwrap(),
                ],
                confidence: 0.85,
                action: Some("system.open_app".to_string()),
            },
            IntentPattern {
                name: "set_timer".to_string(),
                patterns: vec![
                    Regex::new(r"поставь\s+таймер\s+на\s+(\d+)\s+(минут|секунд|час)").unwrap(),
                    Regex::new(r"таймер\s+на\s+(\d+)\s+(минут|секунд|час)").unwrap(),
                ],
                confidence: 0.9,
                action: Some("assistant.timer".to_string()),
            },
            IntentPattern {
                name: "weather".to_string(),
                patterns: vec![
                    Regex::new(r"какая\s+погода").unwrap(),
                    Regex::new(r"погода\s+(сегодня|завтра|сейчас)").unwrap(),
                ],
                confidence: 0.85,
                action: Some("assistant.weather".to_string()),
            },
        ])
    }

    fn create_entity_extractors() -> HashMap<String, EntityExtractor> {
        let mut extractors = HashMap::new();

        extractors.insert(
            "room".to_string(),
            EntityExtractor {
                name: "room".to_string(),
                pattern: Regex::new(r"в\s+(кухне|спальне|гостиной|ванной|комнате|зале|коридоре)")
                    .unwrap(),
                entity_type: EntityType::Room,
            },
        );

        extractors.insert(
            "number".to_string(),
            EntityExtractor {
                name: "number".to_string(),
                pattern: Regex::new(r"\d+").unwrap(),
                entity_type: EntityType::Number,
            },
        );

        extractors.insert(
            "app_name".to_string(),
            EntityExtractor {
                name: "app_name".to_string(),
                pattern: Regex::new(
                    r"(блокнот|браузер|калькулятор|телеграм|discord|spotify|chrome|firefox)",
                )
                .unwrap(),
                entity_type: EntityType::Custom("app".to_string()),
            },
        );

        extractors
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        for extractor in self.entities.values() {
            if let Some(mat) = extractor.pattern.find(text) {
                entities.push(Entity {
                    name: extractor.name.clone(),
                    value: serde_json::Value::String(mat.as_str().to_string()),
                    start: mat.start(),
                    end: mat.end(),
                });
            }
        }

        entities
    }
}

#[async_trait]
impl NluProcessor for PatternMatcher {
    async fn process(&self, text: &str) -> KlarnetResult<NluResult> {
        let lower_text = text.to_lowercase();

        // Find matching patterns
        for pattern in &self.patterns {
            for regex in &pattern.patterns {
                if regex.is_match(&lower_text) {
                    let entities = self.extract_entities(&lower_text);

                    let mut parameters = serde_json::Map::new();
                    for entity in &entities {
                        parameters.insert(entity.name.clone(), entity.value.clone());
                    }

                    return Ok(NluResult {
                        transcript: text.to_string(),
                        intent: Some(Intent {
                            name: pattern.name.clone(),
                            confidence: pattern.confidence,
                            entities: entities.clone(),
                        }),
                        wake_word_detected: false,
                        command_type: CommandType::Local(LocalCommand {
                            action: pattern.action.clone().unwrap_or(pattern.name.clone()),
                            parameters,
                        }),
                    });
                }
            }
        }

        // No pattern matched
        Ok(NluResult {
            transcript: text.to_string(),
            intent: None,
            wake_word_detected: false,
            command_type: CommandType::Unknown,
        })
    }

    fn name(&self) -> &str {
        "PatternMatcher"
    }
}
