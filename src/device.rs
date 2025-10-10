#[cfg(feature = "hardware")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hardware")]
use std::collections::HashSet;
#[cfg(feature = "hardware")]
use std::io::{self, IsTerminal, Write};
#[cfg(feature = "hardware")]
use tracing::{info, warn};

use tracing::debug;
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
                config.audio().input_device.as_deref(),
            )? {
                config.audio_mut().input_device = Some(choice);
            }
        } else {
            warn!("CPAL did not report any input devices");
        }

        if !output_devices.is_empty() {
            let current = config
                .tts()
                .device
                .as_deref()
                .or(config.audio().output_device.as_deref());
            if let Some(choice) = prompt_for_device("вывода", &output_devices, current)? {
                config.tts_mut().device = Some(choice.clone());
                config.audio_mut().output_device = Some(choice);
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

pub fn force_interactive_device_selection(config: &mut AppConfig) -> Result<()> {
    #[cfg(feature = "hardware")]
    {
        println!("\n=== Настройка аудио устройств KLARNET ===\n");

        // Получаем список устройств
        let input_devices = enumerate_input_devices()?;
        let output_devices = enumerate_output_devices()?;

        // Показываем текущие настройки
        println!("Текущие настройки:");
        println!("  Устройство ввода: {}",
                 config.audio().input_device.as_deref().unwrap_or("по умолчанию"));
        println!("  Устройство вывода: {}\n",
                 config.tts().device.as_deref().unwrap_or("по умолчанию"));

        // Выбор устройства ввода (микрофона)
        if !input_devices.is_empty() {
            println!("📤 Устройства ввода (микрофоны):");
            for (idx, name) in input_devices.iter().enumerate() {
                println!("  [{}] {}", idx + 1, name);
            }
            println!("  [0] Использовать системное устройство по умолчанию");

            loop {
                print!("\nВыберите устройство ввода (0-{}): ", input_devices.len());
                io::stdout().flush()?;

                let mut buffer = String::new();
                io::stdin().read_line(&mut buffer)?;
                let trimmed = buffer.trim();

                if let Ok(choice) = trimmed.parse::<usize>() {
                    if choice == 0 {
                        config.audio_mut().input_device = None;
                        println!("✓ Используется устройство ввода по умолчанию");
                        break;
                    } else if choice > 0 && choice <= input_devices.len() {
                        let selected = input_devices[choice - 1].clone();
                        config.audio_mut().input_device = Some(selected.clone());
                        println!("✓ Выбрано устройство ввода: {}", selected);
                        break;
                    }
                }
                println!("❌ Некорректный выбор. Попробуйте снова.");
            }
        } else {
            warn!("Не найдено ни одного устройства ввода");
        }

        println!();

        // Выбор устройства вывода (динамики/наушники)
        if !output_devices.is_empty() {
            println!("🔊 Устройства вывода (динамики/наушники):");
            for (idx, name) in output_devices.iter().enumerate() {
                // Помечаем популярные устройства
                let mut label = name.clone();
                if name.to_lowercase().contains("headphone") {
                    label.push_str(" 🎧");
                } else if name.to_lowercase().contains("speaker") {
                    label.push_str(" 🔊");
                }
                println!("  [{}] {}", idx + 1, label);
            }
            println!("  [0] Использовать системное устройство по умолчанию");

            loop {
                print!("\nВыберите устройство вывода (0-{}): ", output_devices.len());
                io::stdout().flush()?;

                let mut buffer = String::new();
                io::stdin().read_line(&mut buffer)?;
                let trimmed = buffer.trim();

                if let Ok(choice) = trimmed.parse::<usize>() {
                    if choice == 0 {
                        config.tts_mut().device = None;
                        config.audio_mut().output_device = None;
                        println!("✓ Используется устройство вывода по умолчанию");
                        break;
                    } else if choice > 0 && choice <= output_devices.len() {
                        let selected = output_devices[choice - 1].clone();
                        config.tts_mut().device = Some(selected.clone());
                        config.audio_mut().output_device = Some(selected.clone());
                        println!("✓ Выбрано устройство вывода: {}", selected);
                        break;
                    }
                }
                println!("❌ Некорректный выбор. Попробуйте снова.");
            }
        } else {
            warn!("Не найдено ни одного устройства вывода");
        }

        println!("\n=== Настройка завершена ===\n");

        // Логируем итоговую конфигурацию
        info!(
            "Аудио конфигурация: вход={}, выход={}",
            config.audio().input_device.as_deref().unwrap_or("default"),
            config.tts().device.as_deref().unwrap_or("default")
        );

        Ok(())
    }

    #[cfg(not(feature = "hardware"))]
    {
        let _ = config;
        println!("Hardware feature не включена. Используются настройки по умолчанию.");
        Ok(())
    }
}

// Добавьте функцию для тестирования аудио устройств
pub async fn test_audio_devices(config: &AppConfig) -> Result<()> {
    #[cfg(feature = "hardware")]
    {
        use std::time::Duration;
        use tokio::time::sleep;

        println!("\n🔊 Тестирование аудио устройств...\n");

        // Генерируем тестовый звук (простой тон)
        let sample_rate = 48000;
        let duration_secs = 0.5;
        let frequency = 440.0; // Нота Ля
        let samples_count = (sample_rate as f32 * duration_secs) as usize;

        let mut test_sound = Vec::with_capacity(samples_count * 2);
        for i in 0..samples_count {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
            let sample_i16 = (sample * 0.3 * i16::MAX as f32) as i16;
            test_sound.extend_from_slice(&sample_i16.to_le_bytes());
        }

        // Попробуем воспроизвести тестовый звук
        println!("Воспроизведение тестового сигнала (440 Гц)...");

        use crate::app::AppConfig;
        use tts::player::AudioPlayer;

        match AudioPlayer::new(config.tts().device.as_deref()) {
            Ok(player) => {
                match player.play_pcm(&test_sound, sample_rate) {
                    Ok(_) => {
                        println!("✓ Тестовый звук воспроизведён успешно");
                        println!("  Вы должны были услышать короткий тон.\n");

                        // Спрашиваем подтверждение
                        print!("Вы услышали звук? (да/нет/повтор) [да]: ");
                        io::stdout().flush()?;

                        let mut response = String::new();
                        io::stdin().read_line(&mut response)?;
                        let response = response.trim().to_lowercase();

                        match response.as_str() {
                            "нет" | "n" | "no" => {
                                println!("\n⚠️  Если вы не услышали звук, проверьте:");
                                println!("  • Громкость системы");
                                println!("  • Правильность выбранного устройства вывода");
                                println!("  • Подключение динамиков/наушников\n");
                            }
                            "повтор" | "r" | "repeat" => {
                                // Рекурсивно вызываем тест
                                return Box::pin(test_audio_devices(config)).await;
                            }
                            _ => {
                                println!("✓ Отлично! Аудио система работает корректно.\n");
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Ошибка воспроизведения: {}", e);
                        println!("   Проверьте выбранное устройство вывода.\n");
                    }
                }
            }
            Err(e) => {
                println!("❌ Не удалось инициализировать аудио плеер: {}", e);
            }
        }

        sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    #[cfg(not(feature = "hardware"))]
    {
        let _ = config;
        println!("Тестирование аудио недоступно (hardware feature отключена)");
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
