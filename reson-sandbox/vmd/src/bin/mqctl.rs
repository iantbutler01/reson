use anyhow::Result;
use clap::{Parser, Subcommand};
use vmd_rs::config::{Config, ControlBusConfig};
use vmd_rs::control_bus;

#[derive(Parser, Debug)]
#[command(name = "mqctl", about = "Reson sandbox MQ control utilities")]
struct Cli {
    #[arg(long)]
    nats_url: Option<String>,
    #[arg(long)]
    subject_prefix: Option<String>,
    #[arg(long)]
    stream_name: Option<String>,
    #[arg(long)]
    dead_letter_subject: Option<String>,
    #[arg(long)]
    replay_subject: Option<String>,
    #[arg(long)]
    durable: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    ReplayDlq {
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = Config::default();
    let mut control = cfg.control_bus.take().unwrap_or_else(default_control_config);

    if let Some(nats_url) = cli.nats_url {
        control.nats_url = nats_url;
    }

    if let Some(subject_prefix) = cli.subject_prefix {
        let previous_prefix = control.subject_prefix.clone();
        control.subject_prefix = subject_prefix;

        let default_dead_letter = format!("{}.dlq.commands", previous_prefix);
        if control.dead_letter_subject == default_dead_letter {
            control.dead_letter_subject = format!("{}.dlq.commands", control.subject_prefix);
        }
        let default_replay = format!("{}.replay.commands", previous_prefix);
        if control.replay_subject == default_replay {
            control.replay_subject = format!("{}.replay.commands", control.subject_prefix);
        }
    }

    if let Some(stream_name) = cli.stream_name {
        control.stream_name = stream_name;
    }
    if let Some(dead_letter_subject) = cli.dead_letter_subject {
        control.dead_letter_subject = dead_letter_subject;
    }
    if let Some(replay_subject) = cli.replay_subject {
        control.replay_subject = replay_subject;
    }
    if let Some(durable) = cli.durable {
        control.command_consumer_durable = durable;
    }

    match cli.command {
        Commands::ReplayDlq { limit } => {
            let replayed = control_bus::replay_dead_letters(control, limit).await?;
            println!("replayed_dead_letters={replayed}");
        }
    }

    Ok(())
}

fn default_control_config() -> ControlBusConfig {
    let subject_prefix = "reson.sandbox.control".to_string();
    ControlBusConfig {
        nats_url: "nats://127.0.0.1:4222".to_string(),
        subject_prefix: subject_prefix.clone(),
        node_id: "mqctl".to_string(),
        dedupe_etcd_endpoints: Vec::new(),
        dedupe_prefix: "/reson-sandbox/command-dedupe".to_string(),
        stream_name: "RESON_SANDBOX_CONTROL".to_string(),
        stream_max_age_secs: 60 * 60 * 24 * 7,
        stream_replicas: 1,
        command_consumer_durable: "mqctl-replay".to_string(),
        command_max_deliver: 5,
        command_ack_wait_ms: 30_000,
        dead_letter_subject: format!("{subject_prefix}.dlq.commands"),
        replay_subject: format!("{subject_prefix}.replay.commands"),
    }
}
