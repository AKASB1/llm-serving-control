# Security

Please report a security problem privately through GitHub: on the repository page, open **Security**, then **Report a vulnerability**. If that button is missing, open an issue that asks for a private contact, without details of the problem. Do not put exploit details in a public issue.

The live proxy (`cmd/control-plane`) binds to 127.0.0.1 by default (`deploy/README.md`), and its admin endpoints have no authentication. Do not expose it to an untrusted network.
