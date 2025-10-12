use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use klarnet_core::{KlarnetError, KlarnetResult};
use phonetic::PhoneticMatcher;
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value};

#[derive(Debug, Deserialize)]
struct RawPatternsConfig {
    intents: Vec<RawPhoneticIntent>,
}

#[derive(Debug, Deserialize)]
struct RawPhoneticIntent {
    name: String,
    #[serde(default)]
    patterns: Vec<String>,
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    params: BTreeMap<String, Value>,
    #[serde(default)]
    requires_llm: bool,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct PhoneticIntentMetadata {
    pub name: String,
    pub action: String,
    pub parameters: JsonMap<String, Value>,
    pub requires_llm: bool,
    pub base_confidence: f32,
}

pub struct LoadedPhoneticPatterns {
    pub matcher: PhoneticMatcher,
    pub metadata: HashMap<String, PhoneticIntentMetadata>,
}

pub fn load_patterns(path: &Path) -> KlarnetResult<LoadedPhoneticPatterns> {
    let data = fs::read_to_string(path).map_err(|err| {
        KlarnetError::Config(format!(
            "Failed to read phonetic patterns file {}: {err}",
            path.display()
        ))
    })?;
    let config: RawPatternsConfig = serde_yaml::from_str(&data).map_err(|err| {
        KlarnetError::Config(format!(
            "Failed to parse phonetic patterns file {}: {err}",
            path.display()
        ))
    })?;

    let mut matcher = PhoneticMatcher::new();
    let mut metadata = HashMap::new();

    for intent in config.intents {
        if intent.patterns.is_empty() {
            continue;
        }

        let action = intent.intent.unwrap_or_else(|| intent.name.clone());
        let confidence = intent.confidence.unwrap_or(0.9);
        let mut params = JsonMap::new();
        for (key, value) in intent.params.into_iter() {
            params.insert(key, value);
        }

        for pattern in intent.patterns {
            let pattern = pattern.trim();
            if pattern.is_empty() {
                continue;
            }

            matcher.add_command(pattern, &intent.name, confidence);
            metadata.insert(
                pattern.to_string(),
                PhoneticIntentMetadata {
                    name: intent.name.clone(),
                    action: action.clone(),
                    parameters: params.clone(),
                    requires_llm: intent.requires_llm,
                    base_confidence: confidence,
                },
            );
        }
    }

    Ok(LoadedPhoneticPatterns { matcher, metadata })
}

/// Helper exposed for tests.
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_patterns_from_yaml() {
        let mut file = NamedTempFile::new().expect("temp file");
        let yaml = r#"
intents:
  - name: test
    patterns:
      - "включи свет"
    intent: "smart_home.lights"
    params:
      state: "on"
"#;
        std::io::Write::write_all(&mut file, yaml.as_bytes()).expect("write");

        let loaded = load_patterns(file.path()).expect("load patterns");
        assert_eq!(loaded.metadata.len(), 1);
        let entry = loaded.metadata.get("включи свет").expect("metadata entry");
        assert_eq!(entry.action, "smart_home.lights");
        assert_eq!(entry.parameters.get("state").unwrap(), "on");
        assert!(loaded.matcher.match_text_within("фключи сфет", 2).is_some());
    }
}