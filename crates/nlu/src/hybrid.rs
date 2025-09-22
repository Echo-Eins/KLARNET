// crates/nlu/src/hybrid.rs

pub struct HybridNlu {
    pattern_matcher: PatternMatcher,
    llm_processor: LlmProcessor,
    confidence_threshold: f32,
}

impl HybridNlu {
    pub async fn new(config: NluConfig) -> KlarnetResult<Self> {
        Ok(Self {
            pattern_matcher: PatternMatcher::new(config.patterns_file.clone()).await?,
            llm_processor: LlmProcessor::new(config.llm_config.clone()).await?,
            confidence_threshold: config.confidence_threshold,
        })
    }
}

#[async_trait]
impl NluProcessor for HybridNlu {
    async fn process(&self, text: &str) -> KlarnetResult<NluResult> {
        // First try pattern matching
        let local_result = self.pattern_matcher.process(text).await?;

        if let Some(intent) = &local_result.intent {
            if intent.confidence >= self.confidence_threshold {
                debug!("Using local pattern match: {}", intent.name);
                return Ok(local_result);
            }
        }

        // Fall back to LLM
        debug!("Falling back to LLM processing");
        self.llm_processor.process(text).await
    }

    fn name(&self) -> &str {
        "HybridNlu"
    }
}