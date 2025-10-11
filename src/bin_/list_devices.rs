use cpal::traits::{DeviceTrait, HostTrait};
use cpal::SampleFormat;

fn main() {
    let host = cpal::default_host();

    println!("=== INPUT DEVICES ===");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            if let Ok(name) = device.name() {
                println!("\n{}. {}", i + 1, name);

                if let Ok(configs) = device.supported_input_configs() {
                    for config in configs {
                        println!("   Format: {:?}, Channels: {}, Sample Rate: {}-{}",
                                 config.sample_format(),
                                 config.channels(),
                                 config.min_sample_rate().0,
                                 config.max_sample_rate().0
                        );
                    }
                }
            }
        }
    }

    println!("\n=== OUTPUT DEVICES ===");
    if let Ok(devices) = host.output_devices() {
        for (i, device) in devices.enumerate() {
            if let Ok(name) = device.name() {
                println!("\n{}. {}", i + 1, name);

                if let Ok(configs) = device.supported_output_configs() {
                    for config in configs {
                        println!("   Format: {:?}, Channels: {}, Sample Rate: {}-{}",
                                 config.sample_format(),
                                 config.channels(),
                                 config.min_sample_rate().0,
                                 config.max_sample_rate().0
                        );
                    }
                }
            }
        }
    }
}