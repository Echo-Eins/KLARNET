use std::fs;
use std::path::Path;

use regex::{Regex, RegexBuilder};
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value};

use klarnet_core::{Entity, KlarnetError, KlarnetResult};

use std::collections::HashMap;

use crate::{FallbackConfig, LocalNluConfig};

/// Result of matching a local intent pattern.
#[derive(Debug)]
pub(crate) struct MatchOutcome {
    pub intent_name: String,
    pub confidence: f32,
    pub action: Option<String>,
    pub parameters: JsonMap<String, Value>,
    pub entities: Vec<Entity>,
    pub route: Option<String>,
}

#[derive(Debug)]
pub(crate) struct LocalIntentMatcher {
    intents: Vec<CompiledIntent>,
    pub(crate) fallback: FallbackConfig,
}

impl LocalIntentMatcher {
    pub(crate) fn new(config: &LocalNluConfig) -> KlarnetResult<Self> {
        let intents: IntentConfig = load_config(&config.intents_path)?;
        let entities: EntitiesConfig = load_config(&config.entities_path)?;

        let entity_matchers = entities.build_matchers()?;
        let file_fallback = intents.fallback.clone();
        let compiled = intents
            .intents
            .into_iter()
            .map(|intent| intent.compile(&entity_matchers))
            .collect::<KlarnetResult<Vec<_>>>()?;

        let fallback = config
            .fallback
            .clone()
            .or(file_fallback)
            .unwrap_or_default();

        Ok(Self {
            intents: compiled,
            fallback,
        })
    }

    pub(crate) fn match_text(
        &self,
        text: &str,
        offset: usize,
    ) -> KlarnetResult<Option<MatchOutcome>> {
        for intent in &self.intents {
            if let Some(outcome) = intent.match_text(text, offset)? {
                return Ok(Some(outcome));
            }
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct CompiledIntent {
    name: String,
    confidence: f32,
    action: Option<String>,
    route: Option<String>,
    parameters: JsonMap<String, Value>,
    patterns: Vec<CompiledPattern>,
}

impl CompiledIntent {
    fn match_text(&self, text: &str, offset: usize) -> KlarnetResult<Option<MatchOutcome>> {
        for pattern in &self.patterns {
            if let Some(caps) = pattern.regex.captures(text) {
                let mut parameters = self.parameters.clone();
                let mut slot_values: HashMap<String, Value> = HashMap::new();
                let mut entities = Vec::new();

                for slot in &pattern.slots {
                    if let Some(m) = caps.name(&slot.name) {
                        let value = slot.matcher.resolve_value(m.as_str())?;
                        slot_values.insert(slot.name.clone(), value.clone());
                        parameters.insert(slot.name.clone(), value.clone());
                        entities.push(Entity {
                            name: slot.name.clone(),
                            value,
                            start: offset + m.start(),
                            end: offset + m.end(),
                        });
                    }
                }

                for (key, value) in self.parameters.iter() {
                    if let Some(slot_name) = value
                        .as_str()
                        .and_then(|v| v.strip_prefix('{'))
                        .and_then(|v| v.strip_suffix('}'))
                    {
                        if let Some(slot_value) = slot_values.get(slot_name) {
                            parameters.insert(key.clone(), slot_value.clone());
                        }
                    }
                }

                let outcome = MatchOutcome {
                    intent_name: self.name.clone(),
                    confidence: self.confidence,
                    action: self.action.clone(),
                    parameters,
                    entities,
                    route: self.route.clone(),
                };

                return Ok(Some(outcome));
            }
        }

        Ok(None)
    }
}

#[derive(Debug)]
struct CompiledPattern {
    regex: Regex,
    slots: Vec<SlotBinding>,
}

#[derive(Debug, Clone)]
struct SlotBinding {
    name: String,
    matcher: EntityMatcher,
}

#[derive(Debug, Clone)]
enum EntityMatcher {
    List {
        synonyms: HashMap<String, String>,
    },
    Regex {
        pattern: String,
        transform: Option<EntityTransform>,
    },
}

impl EntityMatcher {
    fn regex_fragment(&self) -> KlarnetResult<String> {
        match self {
            EntityMatcher::List { synonyms } => {
                let pattern = synonyms
                    .keys()
                    .map(|key| regex::escape(key))
                    .collect::<Vec<_>>()
                    .join("|");
                Ok(format!("(?:{})", pattern))
            }
            EntityMatcher::Regex { pattern, .. } => Ok(pattern.clone()),
        }
    }

    fn resolve_value(&self, value: &str) -> KlarnetResult<Value> {
        match self {
            EntityMatcher::List { synonyms } => synonyms
                .get(&value.to_lowercase())
                .cloned()
                .map(Value::String)
                .ok_or_else(|| KlarnetError::Nlu(format!("Unknown entity value '{}'", value))),
            EntityMatcher::Regex { transform, .. } => {
                let mut value = Value::String(value.to_string());
                if let Some(transform) = transform {
                    transform.apply(&mut value)?;
                }
                Ok(value)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum EntityTransform {
    Int,
    Float,
    Lowercase,
}

impl EntityTransform {
    fn apply(&self, value: &mut Value) -> KlarnetResult<()> {
        match self {
            EntityTransform::Int => {
                if let Some(text) = value.as_str() {
                    let parsed = text.parse::<i64>().map_err(|err| {
                        KlarnetError::Nlu(format!("Failed to parse integer entity: {}", err))
                    })?;
                    *value = Value::Number(parsed.into());
                }
            }
            EntityTransform::Float => {
                if let Some(text) = value.as_str() {
                    let parsed = text.parse::<f64>().map_err(|err| {
                        KlarnetError::Nlu(format!("Failed to parse float entity: {}", err))
                    })?;
                    *value = serde_json::Number::from_f64(parsed)
                        .map(Value::Number)
                        .ok_or_else(|| KlarnetError::Nlu("Invalid float value".to_string()))?;
                }
            }
            EntityTransform::Lowercase => {
                if let Some(text) = value.as_str() {
                    *value = Value::String(text.to_lowercase());
                }
            }
        }

        Ok(())
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "int" => Some(EntityTransform::Int),
            "float" => Some(EntityTransform::Float),
            "lowercase" => Some(EntityTransform::Lowercase),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct IntentConfig {
    intents: Vec<IntentDefinition>,
    #[serde(default)]
    fallback: Option<FallbackConfig>,
}

#[derive(Debug, Deserialize)]
struct IntentDefinition {
    name: String,
    #[serde(default)]
    patterns: Vec<String>,
    #[serde(default = "default_intent_confidence")]
    confidence: f32,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    parameters: JsonMap<String, Value>,
    #[serde(default)]
    route: Option<String>,
}

fn default_intent_confidence() -> f32 {
    0.8
}

impl IntentDefinition {
    fn compile(self, entities: &HashMap<String, EntityMatcher>) -> KlarnetResult<CompiledIntent> {
        let placeholder_regex = Regex::new(r"\{([a-zA-Z0-9_]+)\}").unwrap();
        let mut patterns = Vec::new();

        for pattern in &self.patterns {
            let mut result = String::new();
            let mut slots = Vec::new();
            let mut last = 0;

            for caps in placeholder_regex.captures_iter(pattern) {
                let m = caps.get(0).unwrap();
                result.push_str(&pattern[last..m.start()]);
                let slot_name = caps.get(1).unwrap().as_str().to_string();
                let matcher = entities.get(&slot_name).ok_or_else(|| {
                    KlarnetError::Nlu(format!(
                        "Unknown entity '{}' referenced in intent '{}'",
                        slot_name, self.name
                    ))
                })?;
                result.push_str(&format!("(?P<{}>{})", slot_name, matcher.regex_fragment()?));
                slots.push(SlotBinding {
                    name: slot_name,
                    matcher: matcher.clone(),
                });
                last = m.end();
            }

            result.push_str(&pattern[last..]);

            let regex = RegexBuilder::new(&format!(r"^\s*{}\s*$", result))
                .case_insensitive(true)
                .unicode(true)
                .build()
                .map_err(|e| {
                    KlarnetError::Nlu(format!(
                        "Invalid pattern '{}' for intent '{}': {}",
                        pattern, self.name, e
                    ))
                })?;

            patterns.push(CompiledPattern { regex, slots });
        }
        Ok(CompiledIntent {
            name: self.name,
            confidence: self.confidence,
            action: self.action,
            route: self.route,
            parameters: self.parameters,
            patterns,
        })
    }
}

#[derive(Debug, Deserialize)]
struct EntitiesConfig {
    entities: HashMap<String, RawEntityDefinition>,
}

impl EntitiesConfig {
    fn build_matchers(&self) -> KlarnetResult<HashMap<String, EntityMatcher>> {
        let mut result = HashMap::new();

        for (name, definition) in &self.entities {
            result.insert(name.clone(), definition.to_matcher()?);
        }

        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
struct RawEntityDefinition {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    values: Option<RawEntityValues>,
    #[serde(default)]
    pattern: Option<String>,
    #[serde(default)]
    transform: Option<String>,
}

impl RawEntityDefinition {
    fn to_matcher(&self) -> KlarnetResult<EntityMatcher> {
        match self.kind.as_str() {
            "list" => {
                let values = self.values.clone().ok_or_else(|| {
                    KlarnetError::Nlu("List entity must provide values".to_string())
                })?;
                let mut synonyms = HashMap::new();
                for (synonym, value) in values.into_pairs() {
                    synonyms.insert(synonym.to_lowercase(), value.clone());
                    synonyms.entry(value.to_lowercase()).or_insert(value);
                }
                Ok(EntityMatcher::List { synonyms })
            }
            "regex" => {
                let pattern = self.pattern.clone().ok_or_else(|| {
                    KlarnetError::Nlu("Regex entity must provide pattern".to_string())
                })?;
                let transform = self
                    .transform
                    .as_deref()
                    .and_then(EntityTransform::from_str);
                Ok(EntityMatcher::Regex { pattern, transform })
            }
            other => Err(KlarnetError::Nlu(format!(
                "Unsupported entity type '{}'",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawEntityValues {
    List(Vec<HashMap<String, String>>),
    Map(HashMap<String, String>),
}

impl RawEntityValues {
    fn into_pairs(self) -> Vec<(String, String)> {
        match self {
            RawEntityValues::List(list) => list
                .into_iter()
                .flat_map(|entry| entry.into_iter())
                .collect(),
            RawEntityValues::Map(map) => map.into_iter().collect(),
        }
    }
}

fn load_config<T: serde::de::DeserializeOwned>(path: &Path) -> KlarnetResult<T> {
    let contents = fs::read_to_string(path)
        .map_err(|e| KlarnetError::Config(format!("Failed to read {}: {}", path.display(), e)))?;

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    if ext.eq_ignore_ascii_case("json") {
        serde_json::from_str(&contents)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse {}: {}", path.display(), e)))
    } else {
        serde_yaml::from_str(&contents)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse {}: {}", path.display(), e)))
    }
}
