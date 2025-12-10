# Devcontainer Hardening Improvements

This document identifies 10 security hardening improvements for the devcontainer configuration. These changes address three primary threat models: malicious AI output containment, multi-tenant isolation between agents, and data exfiltration prevention. Improvements are prioritized for a balanced approach between security and developer experience.

**Reference:** [GitHub Issue #139](https://github.com/mloda-ai/mloda/issues/139)

---

## Priority Matrix

| Priority | Improvement | Ease | Effect | Breakage Risk |
|----------|-------------|------|--------|---------------|
| 1 | Remove curl-pipe-bash | Very Easy | High | None |
| 2 | Add resource limits | Very Easy | Medium | None |
| 3 | Drop dangerous capabilities | Easy | High | Low |
| 4 | Audit volume isolation | Easy | Medium | None |
| 5 | Audit mounted credentials | Easy | Medium | Low |
| 6 | Enable seccomp profile | Medium | High | Medium |
| 7 | Network segmentation | Medium | Medium | Low |
| 8 | User namespace remapping | Hard | High | High (DinD) |
| 9 | Restrict network egress | Hard | High | High |
| 10 | Restrict binary access | Hard | Low | Very High |

---

## Improvement Checklist

### Tier 1: Quick Wins (Easy + Effective)
- [ ] 1. Remove curl-pipe-bash installation pattern → [Section 9](#9-remove-curl-pipe-bash-installation-pattern)
- [ ] 2. Add container resource limits → [Section 3](#3-add-container-resource-limits)
- [ ] 3. Drop dangerous Linux capabilities → [Section 1](#1-drop-dangerous-linux-capabilities)

### Tier 2: Low-Hanging Fruit (Easy + Moderate Effect)
- [ ] 4. Audit volume isolation for shared paths → [Section 7](#7-audit-volume-isolation-for-shared-paths)
- [ ] 5. Audit and minimize mounted credentials → [Section 10](#10-audit-and-minimize-mounted-credentials)

### Tier 3: Worth the Effort (Medium Effort + Good Effect)
- [ ] 6. Enable seccomp security profile → [Section 2](#2-enable-seccomp-security-profile)
- [ ] 7. Implement network segmentation between agents → [Section 6](#6-implement-network-segmentation-between-agents)

### Tier 4: Consider Carefully (High Effort or Breakage Risk)
- [ ] 8. Enable user namespace remapping → [Section 5](#5-enable-user-namespace-remapping)
- [ ] 9. Restrict network egress with allowlist → [Section 8](#8-restrict-network-egress-with-allowlist)
- [ ] 10. Restrict dangerous binary access → [Section 4](#4-restrict-dangerous-binary-access)

*Note: Detailed sections below are grouped by category. Use the checklist above for implementation order.*

---

## 1. Drop Dangerous Linux Capabilities

**File:** `.devcontainer/docker-compose.yml`

Containers run with Docker's default capability set, which includes potentially dangerous capabilities like CAP_NET_RAW (allows packet crafting/sniffing), CAP_MKNOD (allows device file creation), and others. A malicious AI output could exploit these to perform network attacks, escape container boundaries, or interfere with the host system. Explicitly dropping unnecessary capabilities reduces the attack surface available to compromised processes.

**Pros:**
- Significantly reduces attack surface for container escapes
- Prevents network-based attacks like ARP spoofing from within containers
- Low impact on normal development workflows
- Industry standard security practice

**Cons:**
- May break some debugging tools that require elevated capabilities
- Docker-in-Docker feature may require specific capabilities to function
- Need to test thoroughly to identify which capabilities are actually required

---

## 2. Enable Seccomp Security Profile

**File:** `.devcontainer/docker-compose.yml`

No explicit seccomp profile is configured, so containers use Docker's default profile. While the default blocks some dangerous syscalls, a custom profile tailored for Python development could further restrict syscalls like `ptrace` (process debugging), `mount` (filesystem manipulation), and `reboot`. This provides defense-in-depth against AI-generated code attempting privilege escalation or container escape via syscall exploitation.

**Pros:**
- Defense-in-depth against zero-day container escapes
- Can block dangerous syscalls not covered by default profile
- No impact on normal Python development workflows
- Can be customized per-container based on actual needs

**Cons:**
- Custom profiles require careful tuning to avoid breaking legitimate functionality
- Debugging may become more difficult if strace/ptrace is restricted
- Maintenance overhead for keeping profile updated

---

## 3. Add Container Resource Limits

**File:** `.devcontainer/docker-compose.yml`

No CPU or memory limits are defined for agent containers. Malicious or buggy AI-generated code could consume unbounded resources, causing denial-of-service conditions on the host or affecting other agent containers. Resource limits prevent runaway processes from impacting system stability and ensure fair resource allocation between concurrent agents.

**Pros:**
- Prevents single container from monopolizing host resources
- Protects against fork bombs and memory exhaustion attacks
- Ensures predictable performance across all agent containers
- Easy to implement with minimal DX impact

**Cons:**
- May need tuning for resource-intensive operations (large model inference, heavy data processing)
- Could cause legitimate operations to fail if limits are too restrictive
- Adds complexity to troubleshooting performance issues

---

## 4. Restrict Dangerous Binary Access

**File:** `.devcontainer/Dockerfile`

The container includes a full set of system utilities including potentially dangerous binaries like `curl`, `wget`, `nc` (netcat), compilers, and shell interpreters. AI-generated malicious code could leverage these tools to download additional payloads, establish reverse shells, or compile exploits. Removing or restricting access to high-risk binaries limits the tools available to an attacker.

**Pros:**
- Limits ability to download additional malicious payloads
- Prevents easy establishment of reverse shells
- Reduces tools available for lateral movement
- Defense against "living off the land" attacks

**Cons:**
- Significant impact on developer experience (curl/wget commonly needed)
- May break pip, git, and other legitimate tools that shell out to these binaries
- Requires careful analysis of what's actually needed vs. dangerous
- Could be circumvented by AI writing equivalent functionality in Python

---

## 5. Enable User Namespace Remapping

**File:** `.devcontainer/docker-compose.yml`, Docker daemon configuration

While containers run as the `vscode` user internally, this maps to an actual user on the host. User namespace remapping would map container UIDs to unprivileged ranges on the host, so even if an attacker achieves root inside the container, they would still be unprivileged on the host. This adds a strong boundary between container and host privilege levels.

**Pros:**
- Strong defense against container escape leading to host root access
- Industry best practice for multi-tenant container environments
- Transparent to normal container operations
- Works well with non-root container user pattern already in place

**Cons:**
- Requires Docker daemon configuration change (not just compose file)
- May break Docker-in-Docker functionality
- File permission issues with bind-mounted volumes
- More complex debugging when permissions don't match expectations

---

## 6. Implement Network Segmentation Between Agents

**File:** `.devcontainer/docker-compose.yml`

All agent containers share the same Docker network (`dev-network`), allowing unrestricted network communication between them. A compromised agent could attack other agents, intercept their traffic, or use them as pivot points. Isolating agents on separate networks with explicit inter-container communication rules would contain breaches to individual agents.

**Pros:**
- Limits blast radius if one agent is compromised
- Prevents inter-agent network attacks
- Enables fine-grained control over which services can communicate
- Audit trail for inter-container traffic if logging is enabled

**Cons:**
- More complex network configuration to maintain
- May break legitimate inter-agent communication if needed
- Debugging network issues becomes more complex
- Additional overhead for managing multiple networks

---

## 7. Audit Volume Isolation for Shared Paths

**File:** `.devcontainer/docker-compose.yml`

While each agent has a dedicated named volume (`ai_agent_X_data`), all agents bind-mount the same host paths for git configuration (`~/.gitconfig`, `~/.git-credentials`). Additionally, the workspace appears to be mounted similarly across agents. This creates potential for cross-agent information leakage or interference if agents can write to shared locations.

**Pros:**
- Ensures complete isolation of agent working directories
- Prevents accidental or malicious cross-agent data access
- Clearer security boundaries between agents
- Easier to reason about data flows

**Cons:**
- May require duplicating git configuration per agent
- Complicates sharing legitimate artifacts between agents
- More storage usage if workspace is duplicated
- Need to design explicit mechanism for intentional sharing

---

## 8. Restrict Network Egress with Allowlist

**File:** `.devcontainer/docker-compose.yml`, potential firewall rules

Containers have unrestricted outbound network access, allowing AI-generated code to exfiltrate data to arbitrary external servers, download malicious payloads, or establish command-and-control channels. An egress allowlist limiting outbound connections to required services (PyPI, GitHub, specific APIs) would prevent unauthorized data exfiltration.

**Pros:**
- Strong defense against data exfiltration to arbitrary endpoints
- Prevents download of malicious payloads from unknown sources
- Blocks establishment of C2 channels
- Audit capability for all outbound connections

**Cons:**
- Significant maintenance burden to keep allowlist current
- Will break workflows that require ad-hoc external access
- Complex to implement properly (DNS, HTTPS inspection challenges)
- May require proxy infrastructure for proper implementation

---

## 9. Remove Curl-Pipe-Bash Installation Pattern

**File:** `.devcontainer/Dockerfile` (line ~17)

The Dockerfile contains `RUN curl -fsSL https://claude.ai/install.sh | bash` which downloads and executes a remote script at build time. This pattern is vulnerable to supply-chain attacks: a compromised server could serve malicious code, MITM attacks could inject payloads, and there's no verification of script contents. The script contents should be vendored, checksummed, or replaced with a more secure installation method.

**Pros:**
- Eliminates supply-chain attack vector during image builds
- Build becomes reproducible and auditable
- No dependency on external server availability during builds
- Follows security best practices for container builds

**Cons:**
- Requires manual updates when upstream changes
- Need to vendor and maintain installation script
- May miss automatic updates/improvements from upstream
- Extra maintenance overhead

---

## 10. Audit and Minimize Mounted Credentials

**File:** `.devcontainer/docker-compose.yml`

Git credentials (`~/.gitconfig`, `~/.git-credentials`) are mounted into all agent containers, albeit read-only. If an agent is compromised, these credentials could be read and exfiltrated. Additionally, the credential copying logic in the startup command moves credentials to writable locations. Consider whether all agents need git push access, or if read-only repository access would suffice for most agents.

**Pros:**
- Principle of least privilege for credential access
- Limits damage from credential theft
- Forces explicit consideration of which agents need what access
- Reduces credential sprawl across containers

**Cons:**
- May complicate workflows requiring git push from multiple agents
- Need alternative mechanism for agents that do need write access
- More complex initial setup per-agent
- Credential management becomes more granular (more to manage)

