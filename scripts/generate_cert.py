"""Generate a self-signed TLS certificate for local HTTPS using trustme.

Why this matters:
  Most mobile browsers block getUserMedia() (camera access) over plain HTTP.
  Running the Aegis server with HTTPS — even with a self-signed cert — lifts
  that restriction.  The user accepts a one-time browser warning on the phone.

Usage:
    pip install trustme
    python scripts/generate_cert.py

Writes to certs/ (git-ignored):
    cert.pem  — server certificate chain (present to the browser)
    key.pem   — private key (keep secret, never commit)
    ca.pem    — optional: install on your phone to remove the warning entirely
"""

import pathlib


def main() -> None:
    try:
        import trustme
    except ImportError:
        print("trustme is not installed.  Run:  pip install trustme")
        raise SystemExit(1)

    certs_dir = pathlib.Path(__file__).parent.parent / "certs"
    certs_dir.mkdir(exist_ok=True)

    # trustme creates a tiny local CA then issues a cert signed by it.
    # This is enough for browsers to establish TLS; the CA warning appears
    # because the CA isn't in the system trust store.
    ca = trustme.CA()
    server_cert = ca.issue_cert(
        "localhost",
        "127.0.0.1",
        # Add your local network IP here if you want to avoid the warning
        # after installing ca.pem on your phone, e.g. "192.168.1.100"
    )

    cert_path = certs_dir / "cert.pem"
    key_path  = certs_dir / "key.pem"
    ca_path   = certs_dir / "ca.pem"

    # Write private key
    server_cert.private_key_pem.write_to_path(str(key_path))

    # Write certificate chain (may be multiple blobs)
    cert_path.unlink(missing_ok=True)
    for blob in server_cert.cert_chain_pems:
        blob.write_to_path(str(cert_path), append=True)

    # Write CA cert (optional — install on phone to suppress browser warning)
    ca.cert_pem.write_to_path(str(ca_path))

    print(f"cert.pem  →  {cert_path}")
    print(f"key.pem   →  {key_path}")
    print(f"ca.pem    →  {ca_path}  (install on phone to remove browser warning)")
    print()
    print("Next steps:")
    print("  1. Set TLS_ENABLED=true in .env")
    print("  2. python main.py")
    print("  3. Open https://<server-ip>:8765 on your phone")
    print("  4. Accept the security warning (or install ca.pem to avoid it)")


if __name__ == "__main__":
    main()
