// src/commands.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    SystemControl(SystemCommand),
    SmartHome(SmartHomeCommand),
    Assistant(AssistantCommand),
    Custom(String, serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemCommand {
    OpenApplication { name: String },
    CloseApplication { name: String },
    SetVolume { level: u8 },
    Shutdown,
    Restart,
    Lock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmartHomeCommand {
    LightControl { room: String, action: LightAction },
    TemperatureControl { room: String, temperature: f32 },
    DeviceControl { device: String, action: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightAction {
    On,
    Off,
    Dim { level: u8 },
    Color { rgb: (u8, u8, u8) },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantCommand {
    StartRecording,
    StopRecording,
    SetLanguage { language: String },
    ChangeWakeWord { word: String },
}