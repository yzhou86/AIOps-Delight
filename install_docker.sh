#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
  TARGET_USER="${SUDO_USER:-root}"
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Docker. Re-run as root or install sudo first." >&2
    exit 1
  fi
  SUDO="sudo"
  TARGET_USER="${USER}"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported Linux distribution: /etc/os-release was not found." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

install_with_apt() {
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y ca-certificates curl gnupg
  ${SUDO} install -m 0755 -d /etc/apt/keyrings

  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" | ${SUDO} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  ${SUDO} chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null

  ${SUDO} apt-get update
  ${SUDO} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_with_dnf() {
  ${SUDO} dnf -y install dnf-plugins-core
  ${SUDO} dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
  ${SUDO} dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_with_yum() {
  ${SUDO} yum -y install yum-utils
  ${SUDO} yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
  ${SUDO} yum -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

case "${ID}" in
  ubuntu|debian)
    install_with_apt
    ;;
  centos|rhel|rocky|almalinux|fedora)
    if command -v dnf >/dev/null 2>&1; then
      install_with_dnf
    else
      install_with_yum
    fi
    ;;
  *)
    if [[ "${ID_LIKE:-}" == *debian* ]]; then
      install_with_apt
    elif [[ "${ID_LIKE:-}" == *rhel* ]] || [[ "${ID_LIKE:-}" == *fedora* ]]; then
      if command -v dnf >/dev/null 2>&1; then
        install_with_dnf
      else
        install_with_yum
      fi
    else
      echo "Unsupported Linux distribution: ${PRETTY_NAME}" >&2
      exit 1
    fi
    ;;
esac

${SUDO} systemctl enable --now docker

if id -u "${TARGET_USER}" >/dev/null 2>&1 && [[ "${TARGET_USER}" != "root" ]]; then
  ${SUDO} usermod -aG docker "${TARGET_USER}" || true
fi

echo
echo "Docker installation is complete."
echo "Test it with: docker --version"
echo "If Docker permission is still denied, log out and back in so the docker group change takes effect."
