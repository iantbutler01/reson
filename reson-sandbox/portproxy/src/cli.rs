use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Port proxy server and client")]
pub struct Args {
    /// Run in server mode
    #[arg(long)]
    pub server: bool,

    /// Port for the gRPC server (server mode)
    #[arg(long = "rpc-port", default_value_t = 13_338)]
    pub rpc_port: u16,

    /// Address for the TCP proxy listener (server mode) or remote server (client mode)
    #[arg(long = "server-addr", default_value = "0.0.0.0:13337")]
    pub server_addr: String,

    /// Port to listen on (client mode)
    #[arg(long = "listen-port")]
    pub listen_port: Option<u16>,

    /// Destination port to forward to (client mode)
    #[arg(long = "forward-port")]
    pub forward_port: Option<u16>,
}

impl Args {
    pub fn validate(&self) -> Result<(), String> {
        if self.server {
            return Ok(());
        }
        match (self.listen_port, self.forward_port) {
            (Some(_), Some(_)) => Ok(()),
            _ => Err("Client mode requires --listen-port and --forward-port".into()),
        }
    }
}
