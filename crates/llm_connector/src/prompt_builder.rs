#[derive(Debug, Default, Clone)]
pub struct PromptBuilder {
    system_prompt: String,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            system_prompt: default_system_prompt(),
        }
    }

    pub fn build_command_prompt(&self, user_text: &str) -> String {
        format!(
            "{system}\nUser command: {user}\nRespond with valid JSON using keys: intent, confidence, action (optional), route (optional), parameters (object), response (string).",
            system = self.system_prompt,
            user = user_text.trim(),
        )
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}

fn default_system_prompt() -> String {
    "You are a Russian voice assistant NLU component. Interpret spoken commands, map them to assistant actions and extract structured information.".to_string()
}