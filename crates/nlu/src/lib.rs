use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use klarnet_core::{
    CommandRoute, CommandRouting, CommandType, Entity, Intent, KlarnetError, KlarnetResult,
    LocalCommand, NluResult, Transcript,
};
use regex::{Regex, RegexBuilder};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};
use tracing::{debug, warn};

const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluConfig {
    pub mode: NluMode,
    #[serde(default = "default_wake_words")]
    pub wake_words: Vec<String>,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default)]
    pub local: Option<LocalNluConfig>,
    #[serde(default)]
    pub llm: Option<LlmModeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NluMode {
    Local,
    Llm,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalNluConfig {
    pub intents_path: PathBuf,
    pub entities_path: PathBuf,
    #[serde(default)]
    pub fallback: Option<FallbackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FallbackConfig {
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default)]
    pub parameters: JsonMap<String, Value>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModeConfig {
    pub api_key_env: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_s: u64,
}

impl Default for NluConfig {
    fn default() -> Self {
        Self {
            mode: NluMode::Local,
            wake_words: default_wake_words(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            local: Some(LocalNluConfig::default()),
            llm: None,
        }
    }
}

impl Default for LocalNluConfig {
    fn default() -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            intents_path: base.join("../../config/patterns.yaml"),
            entities_path: base.join("../../config/entities.yaml"),
            fallback: None,
        }
    }
}

impl Default for LlmModeConfig {
    fn default() -> Self {
        Self {
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model: "openrouter/auto".to_string(),
            max_tokens: 256,
            temperature: 0.3,
            timeout_s: 10,
        }
    }
}

fn default_wake_words() -> Vec<String> {
    vec!["кларнет".to_string()]
}

fn default_confidence_threshold() -> f32 {
    DEFAULT_CONFIDENCE_THRESHOLD
}

pub struct NluEngine {
    config: NluConfig,
    wake_words_lower: Vec<String>,
    local_matcher: Option<LocalIntentMatcher>,
    llm_client: Option<LlmClient>,
    fallback: FallbackConfig,
}

impl NluEngine {
    pub async fn new(config: NluConfig) -> KlarnetResult<Self> {
        let wake_words_lower = config
            .wake_words
            .iter()
            .map(|w| w.to_lowercase())
            .collect::<Vec<_>>();

        let local_matcher = if let Some(local_config) = config.local.clone() {
            Some(LocalIntentMatcher::new(&local_config)?)
        } else {
            None
        };

        let fallback = local_matcher
            .as_ref()
            .map(|matcher| matcher.fallback.clone())
            .or_else(|| config.local.as_ref().and_then(|c| c.fallback.clone()))
            .unwrap_or_default();

        let llm_client = if matches!(config.mode, NluMode::Llm | NluMode::Hybrid) {
            if let Some(llm_config) = config.llm.clone() {
                Some(LlmClient::new(llm_config)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            wake_words_lower,
            local_matcher,
            llm_client,
            fallback,
        })
    }

    pub async fn process(&self, transcript: &Transcript) -> KlarnetResult<NluResult> {
        let (wake_word_detected, normalized_command, original_command, offset) =
            self.extract_command_text(&transcript.full_text);

        let normalized_command_trimmed = normalized_command.trim();
        let original_command_trimmed = original_command.trim();

        match self.config.mode {
            NluMode::Local => {
                self.process_local(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                    offset,
                )
                    .await
            }
            NluMode::Llm => {
                self.process_llm(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                )
                    .await
            }
            NluMode::Hybrid => {
                self.process_hybrid(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                    offset,
                )
                    .await
            }
        }
    }

    async fn process_local(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
        offset: usize,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        if normalized_command.is_empty() {
            return Ok(self.fallback_result(transcript, true, normalized_command, "empty_command"));
        }

        if let Some(matcher) = &self.local_matcher {
            if let Some(outcome) = matcher.match_text(normalized_command, offset)? {
                return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
            }
        }

        Ok(self.fallback_result(
            transcript,
            wake_word_detected,
            original_command,
            "no_local_match",
        ))
    }

    async fn process_llm(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        if original_command.is_empty() {
            return Ok(self.fallback_result(transcript, true, normalized_command, "empty_command"));
        }

        if let Some(interp) = self.invoke_llm(original_command).await? {
            return Ok(self.build_llm_result(transcript, wake_word_detected, interp));
        }

        Ok(self.fallback_result(
            transcript,
            wake_word_detected,
            original_command,
            "llm_unavailable",
        ))
    }

    async fn process_hybrid(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
        offset: usize,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        let mut low_confidence_match = None;

        if let Some(matcher) = &self.local_matcher {
            if let Some(outcome) = matcher.match_text(normalized_command, offset)? {
                if outcome.confidence >= self.config.confidence_threshold {
                    debug!("Using local intent {}", outcome.intent_name);
                    return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
                }
                low_confidence_match = Some(outcome);
            }
        }

        if let Some(interp) = self.invoke_llm(original_command).await? {
            debug!("Using LLM interpretation for command");
            return Ok(self.build_llm_result(transcript, wake_word_detected, interp));
        }

        if let Some(outcome) = low_confidence_match {
            debug!(
                "Falling back to low confidence local intent {}",
                outcome.intent_name
            );
            return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
        }

        Ok(self.fallback_result(transcript, wake_word_detected, original_command, "no_match"))
    }

    fn build_local_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        outcome: MatchOutcome,
    ) -> NluResult {
        let action = outcome
            .action
            .clone()
            .unwrap_or_else(|| outcome.intent_name.clone());

        let routing_target = outcome.route.clone().or_else(|| Some(action.clone()));

        let mut parameters = outcome.parameters.clone();
        parameters.insert(
            "transcript".to_string(),
            Value::String(transcript.full_text.clone()),
        );

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: Some(Intent {
                name: outcome.intent_name.clone(),
                confidence: outcome.confidence,
                entities: outcome.entities,
            }),
            wake_word_detected,
            command_type: CommandType::Local(LocalCommand {
                action: action.clone(),
                parameters: parameters.clone(),
            }),
            confidence: outcome.confidence,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Local,
                target: routing_target,
                reason: Some("pattern_match".to_string()),
            },
        }
    }

    fn build_llm_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        interpretation: LlmInterpretation,
    ) -> NluResult {
        let mut parameters = interpretation.parameters.clone();
        parameters.insert(
            "transcript".to_string(),
            Value::String(transcript.full_text.clone()),
        );

        let command_type = if let Some(action) = &interpretation.action {
            CommandType::LlmRequired(action.clone())
        } else {
            CommandType::Unknown
        };

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: interpretation.intent_name.map(|name| Intent {
                name,
                confidence: interpretation.confidence,
                entities: interpretation.entities,
            }),
            wake_word_detected,
            command_type,
            confidence: interpretation.confidence,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Llm,
                target: interpretation.route.or(interpretation.action),
                reason: Some("llm_match".to_string()),
            },
        }
    }

    fn fallback_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        command_text: &str,
        reason: &str,
    ) -> NluResult {
        let mut parameters = self.fallback.parameters.clone();
        if !command_text.is_empty() {
            parameters.insert("query".to_string(), Value::String(command_text.to_string()));
        }

        let reason_text = if let Some(base_reason) = &self.fallback.reason {
            format!("{}:{}", base_reason, reason)
        } else {
            reason.to_string()
        };

        let command_type = if let Some(action) = &self.fallback.action {
            CommandType::LlmRequired(action.clone())
        } else {
            CommandType::Unknown
        };

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: None,
            wake_word_detected,
            command_type,
            confidence: 0.0,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Fallback,
                target: self.fallback.route.clone(),
                reason: Some(reason_text),
            },
        }
    }

    fn extract_command_text(&self, text: &str) -> (bool, String, String, usize) {
        if text.trim().is_empty() {
            return (false, String::new(), String::new(), text.len());
        }

        let trimmed_original = text.trim_start();
        let trimmed_lower = trimmed_original.to_lowercase();

        for wake in &self.wake_words_lower {
            if trimmed_lower.starts_with(wake) {
                let wake_chars = wake.chars().count();
                if !is_boundary(&trimmed_lower, wake_chars) {
                    continue;
                }

                let wake_bytes = char_offset(trimmed_original, wake_chars);
                let remainder = &trimmed_original[wake_bytes..];
                let remainder = remainder.trim_start_matches(|c: char| {
                    c.is_whitespace() || matches!(c, ',' | ':' | ';' | '-' | '–')
                });
                let offset = text.len().saturating_sub(remainder.len());
                let normalized = remainder.to_lowercase();
                return (true, normalized, remainder.to_string(), offset);
            }
        }

        let offset = text.len().saturating_sub(trimmed_original.len());
        (false, trimmed_lower, trimmed_original.to_string(), offset)
    }

    async fn invoke_llm(&self, text: &str) -> KlarnetResult<Option<LlmInterpretation>> {
        if let Some(client) = &self.llm_client {
            match client.interpret(text).await {
                Ok(result) => Ok(Some(result)),
                Err(err) => {
                    warn!("LLM interpretation failed: {}", err);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }
}

fn is_boundary(text: &str, chars: usize) -> bool {
    match text.chars().nth(chars) {
        Some(c) => !c.is_alphanumeric(),
        None => true,
    }
}

fn char_offset(text: &str, chars: usize) -> usize {
    if chars == 0 {
        return 0;
    }
    match text.char_indices().nth(chars) {
        Some((idx, _)) => idx,
        None => text.len(),
    }
}

#[derive(Debug)]
struct MatchOutcome {
    intent_name: String,
    confidence: f32,
    action: Option<String>,
    parameters: JsonMap<String, Value>,
    entities: Vec<Entity>,
    route: Option<String>,
}

#[derive(Debug)]
struct LocalIntentMatcher {
    intents: Vec<CompiledIntent>,
    fallback: FallbackConfig,
}

impl LocalIntentMatcher {
    fn new(config: &LocalNluConfig) -> KlarnetResult<Self> {
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

    fn match_text(&self, text: &str, offset: usize) -> KlarnetResult<Option<MatchOutcome>> {
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

#[derive(Debug, Clone)]
enum EntityTransform {
    Int,
    Float,
    Lowercase,
}

impl EntityMatcher {
    fn regex_fragment(&self) -> KlarnetResult<String> {
        Ok(match self {
            EntityMatcher::List { synonyms } => {
                if synonyms.is_empty() {
                    return Err(KlarnetError::Nlu(
                        "Entity list must contain at least one value".to_string(),
                    ));
                }
                let mut variants = synonyms.keys().cloned().collect::<Vec<_>>();
                variants.sort_by(|a, b| b.len().cmp(&a.len()));
                let escaped = variants
                    .into_iter()
                    .map(|v| regex::escape(&v))
                    .collect::<Vec<_>>()
                    .join("|");
                format!("(?:{})", escaped)
            }
            EntityMatcher::Regex { pattern, .. } => format!("(?:{})", pattern),
        })
    }

    fn resolve_value(&self, raw: &str) -> KlarnetResult<Value> {
        match self {
            EntityMatcher::List { synonyms } => {
                let key = raw.trim().to_lowercase();
                if let Some(value) = synonyms.get(&key) {
                    Ok(Value::String(value.clone()))
                } else {
                    Err(KlarnetError::Nlu(format!("Unknown entity value '{}'", raw)))
                }
            }
            EntityMatcher::Regex { transform, .. } => {
                let trimmed = raw.trim();
                match transform {
                    Some(EntityTransform::Int) => trimmed
                        .parse::<i64>()
                        .map(|v| Value::Number(v.into()))
                        .map_err(|e| {
                            KlarnetError::Nlu(format!(
                                "Failed to parse integer value '{}': {}",
                                trimmed, e
                            ))
                        }),
                    Some(EntityTransform::Float) => trimmed
                        .parse::<f64>()
                        .map(|v| Value::Number(serde_json::Number::from_f64(v).unwrap()))
                        .map_err(|e| {
                            KlarnetError::Nlu(format!(
                                "Failed to parse float value '{}': {}",
                                trimmed, e
                            ))
                        }),
                    Some(EntityTransform::Lowercase) => Ok(Value::String(trimmed.to_lowercase())),
                    None => Ok(Value::String(trimmed.to_string())),
                }
            }
        }
    }
}

impl EntityTransform {
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

fn load_config<T: DeserializeOwned>(path: &Path) -> KlarnetResult<T> {
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

struct LlmClient {
    client: reqwest::Client,
    config: LlmModeConfig,
    system_prompt: String,
}

impl LlmClient {
    fn new(config: LlmModeConfig) -> KlarnetResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_s))
            .build()
            .map_err(|e| KlarnetError::Nlu(format!("Failed to build HTTP client: {}", e)))?;

        let system_prompt = r#"You are a Russian voice assistant NLU module.
Return a JSON object with fields: intent (string), confidence (0..1), parameters (object),
entities (object mapping names to canonical values), action (string, optional), route (string, optional)."#
            .to_string();

        Ok(Self {
            client,
            config,
            system_prompt,
        })
    }

    async fn interpret(&self, text: &str) -> KlarnetResult<LlmInterpretation> {
        let response = self.call_llm(text).await?;
        self.parse_response(text, response)
    }

    async fn call_llm(&self, text: &str) -> KlarnetResult<Value> {
        let api_key = std::env::var(&self.config.api_key_env).map_err(|_| {
            KlarnetError::Nlu(format!(
                "API key not found for env var {}",
                self.config.api_key_env
            ))
        })?;

        let payload = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| KlarnetError::Network(format!("LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(KlarnetError::Network(format!(
                "LLM responded with status {}",
                response.status()
            )));
        }

        response
            .json()
            .await
            .map_err(|e| KlarnetError::Nlu(format!("Failed to parse LLM response: {}", e)))
    }

    fn parse_response(&self, _text: &str, payload: Value) -> KlarnetResult<LlmInterpretation> {
        let content = payload
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| KlarnetError::Nlu("Invalid LLM response format".to_string()))?;

        let parsed: Value = serde_json::from_str(content)
            .map_err(|e| KlarnetError::Nlu(format!("Failed to parse LLM JSON payload: {}", e)))?;

        let parameters = parsed
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(JsonMap::new);

        let entities_value = parsed
            .get("entities")
            .cloned()
            .unwrap_or_else(|| Value::Object(JsonMap::new()));

        let mut entities = Vec::new();
        if let Value::Object(map) = entities_value {
            for (name, value) in map {
                entities.push(Entity {
                    name,
                    value,
                    start: 0,
                    end: 0,
                });
            }
        }

        let intent_name = parsed
            .get("intent")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        let confidence = parsed
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.5) as f32;

        let action = parsed
            .get("action")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        let route = parsed
            .get("route")
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .or_else(|| {
                parsed
                    .get("routing")
                    .and_then(|v| v.get("target"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
            });

        Ok(LlmInterpretation {
            intent_name,
            confidence,
            parameters,
            entities,
            action,
            route,
        })
    }
}

struct LlmInterpretation {
    intent_name: Option<String>,
    confidence: f32,
    parameters: JsonMap<String, Value>,
    entities: Vec<Entity>,
    action: Option<String>,
    route: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::Duration};
    use tempfile::tempdir;
    use uuid::Uuid;

    fn transcript(text: &str) -> Transcript {
        Transcript {
            id: Uuid::new_v4(),
            language: "ru".to_string(),
            segments: Vec::new(),
            full_text: text.to_string(),
            processing_time: Duration::from_millis(0),
        }
    }

    #[tokio::test]
    async fn detects_wake_word_and_matches_light_intent() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет включи свет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert_eq!(result.intent.as_ref().unwrap().name, "lights_on");
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Local));
        assert_eq!(
            result
                .parameters
                .get("state")
                .and_then(Value::as_str)
                .unwrap(),
            "on"
        );
    }

    #[tokio::test]
    async fn extracts_entities_from_slots() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет поставь таймер на 10 минут");
        let result = engine.process(&transcript).await.unwrap();

        let params = &result.parameters;
        assert_eq!(result.intent.as_ref().unwrap().name, "set_timer");
        assert_eq!(params.get("number").unwrap().as_i64().unwrap(), 10);
        assert_eq!(
            params.get("time_unit").and_then(Value::as_str).unwrap(),
            "minutes"
        );
    }

    #[tokio::test]
    async fn handles_phrases_without_wake_word() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("включи свет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(!result.wake_word_detected);
        assert!(result.intent.is_none());
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Fallback));
        assert!(matches!(result.command_type, CommandType::Unknown));
    }

    #[tokio::test]
    async fn falls_back_for_unknown_intent() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет расскажи анекдот");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert!(result.intent.is_none());
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Fallback));
    }

    #[tokio::test]
    async fn loads_configuration_from_json() {
        let dir = tempdir().unwrap();
        let intents_path = dir.path().join("intents.json");
        let entities_path = dir.path().join("entities.json");

        fs::write(
            &intents_path,
            serde_json::json!({
                "intents": [
                    {
                        "name": "greet",
                        "patterns": ["привет"],
                        "confidence": 0.6,
                        "action": "assistant.greet",
                        "parameters": {"response": "hello"}
                    }
                ]
            })
                .to_string(),
        )
            .unwrap();

        fs::write(
            &entities_path,
            serde_json::json!({"entities": {}}).to_string(),
        )
            .unwrap();

        let mut config = NluConfig::default();
        config.mode = NluMode::Local;
        config.wake_words = vec!["ассистент".to_string()];
        config.local = Some(LocalNluConfig {
            intents_path: intents_path.clone(),
            entities_path: entities_path.clone(),
            fallback: None,
        });

        let engine = NluEngine::new(config).await.unwrap();
        let transcript = transcript("Ассистент привет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert_eq!(result.intent.as_ref().unwrap().name, "greet");
        assert_eq!(
            result
                .parameters
                .get("response")
                .and_then(Value::as_str)
                .unwrap(),
            "hello"
        );
    }
}