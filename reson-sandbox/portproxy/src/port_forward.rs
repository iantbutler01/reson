// @dive-file: TCP forwarding server/client implementation used by guest hostfwd proxy behavior.
// @dive-rel: Invoked by portproxy/src/main.rs in server and client modes.
// @dive-rel: Carries port-preface handshake and bidirectional stream piping for forwarded traffic.

use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info};

const PORT_BYTES: usize = 2;

pub async fn run_server(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!("Proxy server listening on {}", addr);

    loop {
        let (socket, peer) = listener.accept().await?;
        tokio::spawn(async move {
            if let Err(err) = handle_server_connection(socket, peer).await {
                error!("server connection error from {}: {}", peer, err);
            }
        });
    }
}

async fn handle_server_connection(mut socket: TcpStream, peer: SocketAddr) -> anyhow::Result<()> {
    let mut port_buf = [0u8; PORT_BYTES];
    // @dive: First two bytes are a fixed port preface selecting in-guest destination before payload forwarding begins.
    socket.read_exact(&mut port_buf).await?;
    let dest_port = u16::from_be_bytes(port_buf);
    let target = format!("127.0.0.1:{dest_port}");
    info!("Forwarding connection from {} to {}", peer, target);

    let target_stream = TcpStream::connect(&target).await?;
    copy_bidirectional(socket, target_stream).await?;
    Ok(())
}

pub async fn run_client(
    listen_port: u16,
    forward_port: u16,
    server_addr: &str,
) -> anyhow::Result<()> {
    let listen_addr = format!("0.0.0.0:{listen_port}");
    let listener = TcpListener::bind(&listen_addr).await?;
    info!(
        "Client listening on {}, forwarding to {} via {}",
        listen_addr, forward_port, server_addr
    );

    loop {
        let (socket, peer) = listener.accept().await?;
        let server_addr = server_addr.to_string();
        tokio::spawn(async move {
            if let Err(err) =
                handle_client_connection(socket, forward_port, &server_addr, peer).await
            {
                error!("client connection error from {}: {}", peer, err);
            }
        });
    }
}

async fn handle_client_connection(
    socket: TcpStream,
    forward_port: u16,
    server_addr: &str,
    peer: SocketAddr,
) -> anyhow::Result<()> {
    info!(
        "Forwarding client connection from {} to port {} via {}",
        peer, forward_port, server_addr
    );
    let mut server = TcpStream::connect(server_addr).await?;

    // @dive: Client writes destination preface once so server can route this stream without any additional control channel.
    let port_bytes = forward_port.to_be_bytes();
    server.write_all(&port_bytes).await?;

    copy_bidirectional(socket, server).await?;
    Ok(())
}

async fn copy_bidirectional(mut a: TcpStream, mut b: TcpStream) -> anyhow::Result<()> {
    let (mut ar, mut aw) = a.split();
    let (mut br, mut bw) = b.split();

    // @dive: Bidirectional copy keeps both half-duplex flows active so interactive streams don't deadlock on one-way backpressure.
    let forward = tokio::io::copy(&mut ar, &mut bw);
    let backward = tokio::io::copy(&mut br, &mut aw);

    match tokio::try_join!(forward, backward) {
        Ok(_) => Ok(()),
        Err(err) => Err(err.into()),
    }
}
