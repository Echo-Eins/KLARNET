#[cfg(feature = "hardware")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hardware")]
use std::collections::HashSet;
#[cfg(feature = "hardware")]
use std::io::{self, IsTerminal, Write};
#[cfg(feature = "hardware")]
use tracing::{info, warn};

use crate::app::AppConfig;

#[cfg(feature = "hardware")]
const SKIP_PROMPT_ENV: &str = "KLARNET_SKIP_DEVICE_PROMPT";

pub fn prepare_audio_devices(config: &mut AppConfig) -> Result<()> {
    #[cfg(feature = "hardware")]
    {
        if should_skip_prompt() {
            info!("Skipping interactive audio device selection (env {SKIP_PROMPT_ENV} is set)");
            return Ok(());
        }

        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            info!("Terminal not interactive; keeping configured audio devices");
            return Ok(());
        }

        let input_devices = enumerate_input_devices()?;
        let output_devices = enumerate_output_devices()?;

        if input_devices.is_empty() && output_devices.is_empty() {
            warn!("No audio devices detected by CPAL; using configuration defaults");
            return Ok(());
        }

        if !input_devices.is_empty() {
            if let Some(choice) = prompt_for_device(
                "ввода",
                &input_devices,
                config.audio.input_device.as_deref(),
            )? {
                config.audio.input_device = Some(choice);
            }
        } else {
            warn!("CPAL did not report any input devices");
        }

        if !output_devices.is_empty() {
            let current = config
                .tts
                .device
                .as_deref()
                .or(config.audio.output_device.as_deref());
            if let Some(choice) = prompt_for_device("вывода", &output_devices, current)? {
                config.tts.device = Some(choice.clone());
                config.audio.output_device = Some(choice);
            }
        } else {
            warn!("CPAL did not report any output devices");
        }

        Ok(())
    }

    #[cfg(not(feature = "hardware"))]
    {
        let _ = config;
        Ok(())
    }
}

#[cfg(feature = "hardware")]
fn should_skip_prompt() -> bool {
    std::env::var(SKIP_PROMPT_ENV)
        .map(|value| value.trim() == "1" || value.to_ascii_lowercase() == "true")
        .unwrap_or(false)
}

#[cfg(feature = "hardware")]
fn enumerate_input_devices() -> Result<Vec<String>> {
    enumerate_devices(true)
}

#[cfg(feature = "hardware")]
fn enumerate_output_devices() -> Result<Vec<String>> {
    enumerate_devices(false)
}

#[cfg(feature = "hardware")]
fn enumerate_devices(is_input: bool) -> Result<Vec<String>> {
    use cpal::traits::HostTrait;

    let mut names = Vec::new();
    let mut seen = HashSet::new();

    for host_id in cpal::available_hosts() {
        let host = cpal::host_from_id(host_id)
            .with_context(|| format!("Failed to initialise audio host {host_id:?}"))?;

        let devices = if is_input {
            host.input_devices()
        } else {
            host.output_devices()
        };

        if let Ok(list) = devices {
            for device in list {
                if let Ok(name) = device.name() {
                    if seen.insert(name.clone()) {
                        names.push(name);
                    }
                }
            }
        }
    }

    if names.is_empty() {
        let host = cpal::default_host();
        let fallback = if is_input {
            host.default_input_device()
        } else {
            host.default_output_device()
        };

        if let Some(device) = fallback {
            if let Ok(name) = device.name() {
                if seen.insert(name.clone()) {
                    names.push(name);
                }
            }
        }
    }

    names.sort();
    Ok(names)
}

#[cfg(feature = "hardware")]
fn prompt_for_device(
    kind: &str,
    devices: &[String],
    current: Option<&str>,
) -> Result<Option<String>> {
    use std::str::FromStr;

    println!("Доступные устройства {}:", kind);
    for (idx, name) in devices.iter().enumerate() {
        println!("  {}. {}", idx + 1, name);
    }

    if let Some(current) = current {
        println!("  0. Оставить текущее значение ({current})");
    } else {
        println!("  0. Использовать значение по умолчанию");
    }

    loop {
        print!("Выберите устройство {} (0-{}): ", kind, devices.len());
        io::stdout().flush().ok();

        let mut buffer = String::new();
        io::stdin()
            .read_line(&mut buffer)
            .context("Не удалось прочитать ввод пользователя")?;
        let trimmed = buffer.trim();

        if trimmed.is_empty() {
            println!("Пустой ввод. Попробуйте снова.");
            continue;
        }

        if let Ok(choice) = usize::from_str(trimmed) {
            if choice == 0 {
                return Ok(None);
            }

            if (1..=devices.len()).contains(&choice) {
                let selected = devices[choice - 1].clone();
                info!("Пользователь выбрал устройство {}: '{}'", kind, selected);
                return Ok(Some(selected));
            }
        }

        println!(
            "Некорректный выбор '{trimmed}'. Введите число от 0 до {}.",
            devices.len()
        );
    }
}