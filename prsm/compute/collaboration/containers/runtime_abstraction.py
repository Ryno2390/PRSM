"""
Container Runtime Abstraction Layer for PRSM

This module provides a unified interface for multiple container runtimes,
allowing PRSM to work with Docker, Podman, containerd, and other OCI-compatible runtimes.
This ensures flexibility, security, and future-proofing for enterprise deployments.
"""

import asyncio
import logging
import subprocess
import shutil
import json
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Container configuration that works across all runtimes"""
    image: str
    name: Optional[str] = None
    command: Optional[Union[str, List[str]]] = None
    environment: Optional[Dict[str, str]] = None
    volumes: Optional[Dict[str, str]] = None
    ports: Optional[Dict[str, str]] = None
    working_dir: Optional[str] = None
    user: Optional[str] = None
    network: Optional[str] = None
    security_opts: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None
    detach: bool = True
    remove: bool = False


@dataclass
class ContainerInfo:
    """Container information returned by runtime"""
    id: str
    name: str
    image: str
    status: str
    ports: Dict[str, str]
    created: str
    runtime: str


class ContainerRuntime(ABC):
    """Abstract base class for container runtimes"""
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this runtime is available on the system"""
        pass
    
    @abstractmethod
    async def run_container(self, config: ContainerConfig) -> str:
        """Run a container and return container ID"""
        pass
    
    @abstractmethod
    async def stop_container(self, container_id: str) -> bool:
        """Stop a running container"""
        pass
    
    @abstractmethod
    async def remove_container(self, container_id: str) -> bool:
        """Remove a container"""
        pass
    
    @abstractmethod
    async def list_containers(self, all_containers: bool = False) -> List[ContainerInfo]:
        """List containers"""
        pass
    
    @abstractmethod
    async def get_container_logs(self, container_id: str) -> str:
        """Get container logs"""
        pass
    
    @abstractmethod
    async def build_image(self, dockerfile_path: str, tag: str, context_path: str = ".") -> bool:
        """Build an image from Dockerfile"""
        pass
    
    @abstractmethod
    async def pull_image(self, image: str) -> bool:
        """Pull an image from registry"""
        pass


class DockerRuntime(ContainerRuntime):
    """Docker container runtime implementation"""
    
    def __init__(self):
        self.runtime_name = "docker"
    
    async def is_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = await self._run_command(["docker", "--version"])
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    async def run_container(self, config: ContainerConfig) -> str:
        """Run container with Docker"""
        cmd = ["docker", "run"]
        
        if config.detach:
            cmd.append("-d")
        
        if config.remove:
            cmd.append("--rm")
        
        if config.name:
            cmd.extend(["--name", config.name])
        
        if config.user:
            cmd.extend(["--user", config.user])
        
        if config.working_dir:
            cmd.extend(["--workdir", config.working_dir])
        
        if config.network:
            cmd.extend(["--network", config.network])
        
        # Environment variables
        if config.environment:
            for key, value in config.environment.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Volume mounts
        if config.volumes:
            for host_path, container_path in config.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Port mappings
        if config.ports:
            for host_port, container_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Security options
        if config.security_opts:
            for opt in config.security_opts:
                cmd.extend(["--security-opt", opt])
        
        # Labels
        if config.labels:
            for key, value in config.labels.items():
                cmd.extend(["--label", f"{key}={value}"])
        
        cmd.append(config.image)
        
        if config.command:
            if isinstance(config.command, str):
                cmd.extend(config.command.split())
            else:
                cmd.extend(config.command)
        
        result = await self._run_command(cmd)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Failed to run container: {result.stderr}")
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop Docker container"""
        result = await self._run_command(["docker", "stop", container_id])
        return result.returncode == 0
    
    async def remove_container(self, container_id: str) -> bool:
        """Remove Docker container"""
        result = await self._run_command(["docker", "rm", container_id])
        return result.returncode == 0
    
    async def list_containers(self, all_containers: bool = False) -> List[ContainerInfo]:
        """List Docker containers"""
        cmd = ["docker", "ps", "--format", "json"]
        if all_containers:
            cmd.append("-a")
        
        result = await self._run_command(cmd)
        containers = []
        
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    containers.append(ContainerInfo(
                        id=data.get('ID', ''),
                        name=data.get('Names', ''),
                        image=data.get('Image', ''),
                        status=data.get('Status', ''),
                        ports=self._parse_ports(data.get('Ports', '')),
                        created=data.get('CreatedAt', ''),
                        runtime='docker'
                    ))
        
        return containers
    
    async def get_container_logs(self, container_id: str) -> str:
        """Get Docker container logs"""
        result = await self._run_command(["docker", "logs", container_id])
        return result.stdout
    
    async def build_image(self, dockerfile_path: str, tag: str, context_path: str = ".") -> bool:
        """Build Docker image"""
        cmd = ["docker", "build", "-f", dockerfile_path, "-t", tag, context_path]
        result = await self._run_command(cmd)
        return result.returncode == 0
    
    async def pull_image(self, image: str) -> bool:
        """Pull Docker image"""
        result = await self._run_command(["docker", "pull", image])
        return result.returncode == 0
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    def _parse_ports(self, ports_str: str) -> Dict[str, str]:
        """Parse Docker ports string into dict"""
        ports = {}
        if ports_str:
            # Simple parsing for common port formats
            # This could be enhanced for more complex port mappings
            parts = ports_str.split(', ')
            for part in parts:
                if '->' in part:
                    host, container = part.split('->')
                    ports[host.strip()] = container.strip()
        return ports


class PodmanRuntime(ContainerRuntime):
    """Podman container runtime implementation"""
    
    def __init__(self):
        self.runtime_name = "podman"
    
    async def is_available(self) -> bool:
        """Check if Podman is available"""
        try:
            result = await self._run_command(["podman", "--version"])
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    async def run_container(self, config: ContainerConfig) -> str:
        """Run container with Podman (very similar to Docker)"""
        cmd = ["podman", "run"]
        
        if config.detach:
            cmd.append("-d")
        
        if config.remove:
            cmd.append("--rm")
        
        if config.name:
            cmd.extend(["--name", config.name])
        
        if config.user:
            cmd.extend(["--user", config.user])
        
        if config.working_dir:
            cmd.extend(["--workdir", config.working_dir])
        
        if config.network:
            cmd.extend(["--network", config.network])
        
        # Environment variables
        if config.environment:
            for key, value in config.environment.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Volume mounts
        if config.volumes:
            for host_path, container_path in config.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Port mappings
        if config.ports:
            for host_port, container_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Security options
        if config.security_opts:
            for opt in config.security_opts:
                cmd.extend(["--security-opt", opt])
        
        # Labels
        if config.labels:
            for key, value in config.labels.items():
                cmd.extend(["--label", f"{key}={value}"])
        
        cmd.append(config.image)
        
        if config.command:
            if isinstance(config.command, str):
                cmd.extend(config.command.split())
            else:
                cmd.extend(config.command)
        
        result = await self._run_command(cmd)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Failed to run container: {result.stderr}")
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop Podman container"""
        result = await self._run_command(["podman", "stop", container_id])
        return result.returncode == 0
    
    async def remove_container(self, container_id: str) -> bool:
        """Remove Podman container"""
        result = await self._run_command(["podman", "rm", container_id])
        return result.returncode == 0
    
    async def list_containers(self, all_containers: bool = False) -> List[ContainerInfo]:
        """List Podman containers"""
        cmd = ["podman", "ps", "--format", "json"]
        if all_containers:
            cmd.append("-a")
        
        result = await self._run_command(cmd)
        containers = []
        
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    containers.append(ContainerInfo(
                        id=data.get('Id', ''),
                        name=data.get('Names', [''])[0] if data.get('Names') else '',
                        image=data.get('Image', ''),
                        status=data.get('Status', ''),
                        ports=self._parse_ports_podman(data.get('Ports', [])),
                        created=str(data.get('Created', '')),
                        runtime='podman'
                    ))
        
        return containers
    
    async def get_container_logs(self, container_id: str) -> str:
        """Get Podman container logs"""
        result = await self._run_command(["podman", "logs", container_id])
        return result.stdout
    
    async def build_image(self, dockerfile_path: str, tag: str, context_path: str = ".") -> bool:
        """Build Podman image"""
        cmd = ["podman", "build", "-f", dockerfile_path, "-t", tag, context_path]
        result = await self._run_command(cmd)
        return result.returncode == 0
    
    async def pull_image(self, image: str) -> bool:
        """Pull Podman image"""
        result = await self._run_command(["podman", "pull", image])
        return result.returncode == 0
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    def _parse_ports_podman(self, ports_list: List[Dict]) -> Dict[str, str]:
        """Parse Podman ports list into dict"""
        ports = {}
        for port_info in ports_list:
            host_port = port_info.get('host_port', '')
            container_port = port_info.get('container_port', '')
            if host_port and container_port:
                ports[str(host_port)] = str(container_port)
        return ports


class ContainerdRuntime(ContainerRuntime):
    """Containerd runtime implementation using nerdctl"""
    
    def __init__(self):
        self.runtime_name = "containerd"
    
    async def is_available(self) -> bool:
        """Check if nerdctl (containerd CLI) is available"""
        try:
            result = await self._run_command(["nerdctl", "--version"])
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    async def run_container(self, config: ContainerConfig) -> str:
        """Run container with nerdctl"""
        cmd = ["nerdctl", "run"]
        
        if config.detach:
            cmd.append("-d")
        
        if config.remove:
            cmd.append("--rm")
        
        if config.name:
            cmd.extend(["--name", config.name])
        
        if config.user:
            cmd.extend(["--user", config.user])
        
        if config.working_dir:
            cmd.extend(["--workdir", config.working_dir])
        
        if config.network:
            cmd.extend(["--network", config.network])
        
        # Environment variables
        if config.environment:
            for key, value in config.environment.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Volume mounts
        if config.volumes:
            for host_path, container_path in config.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Port mappings
        if config.ports:
            for host_port, container_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Labels
        if config.labels:
            for key, value in config.labels.items():
                cmd.extend(["--label", f"{key}={value}"])
        
        cmd.append(config.image)
        
        if config.command:
            if isinstance(config.command, str):
                cmd.extend(config.command.split())
            else:
                cmd.extend(config.command)
        
        result = await self._run_command(cmd)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Failed to run container: {result.stderr}")
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop containerd container"""
        result = await self._run_command(["nerdctl", "stop", container_id])
        return result.returncode == 0
    
    async def remove_container(self, container_id: str) -> bool:
        """Remove containerd container"""
        result = await self._run_command(["nerdctl", "rm", container_id])
        return result.returncode == 0
    
    async def list_containers(self, all_containers: bool = False) -> List[ContainerInfo]:
        """List containerd containers"""
        cmd = ["nerdctl", "ps", "--format", "json"]
        if all_containers:
            cmd.append("-a")
        
        result = await self._run_command(cmd)
        containers = []
        
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    containers.append(ContainerInfo(
                        id=data.get('ID', ''),
                        name=data.get('Names', ''),
                        image=data.get('Image', ''),
                        status=data.get('Status', ''),
                        ports=self._parse_ports(data.get('Ports', '')),
                        created=data.get('CreatedAt', ''),
                        runtime='containerd'
                    ))
        
        return containers
    
    async def get_container_logs(self, container_id: str) -> str:
        """Get containerd container logs"""
        result = await self._run_command(["nerdctl", "logs", container_id])
        return result.stdout
    
    async def build_image(self, dockerfile_path: str, tag: str, context_path: str = ".") -> bool:
        """Build containerd image"""
        cmd = ["nerdctl", "build", "-f", dockerfile_path, "-t", tag, context_path]
        result = await self._run_command(cmd)
        return result.returncode == 0
    
    async def pull_image(self, image: str) -> bool:
        """Pull containerd image"""
        result = await self._run_command(["nerdctl", "pull", image])
        return result.returncode == 0
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    def _parse_ports(self, ports_str: str) -> Dict[str, str]:
        """Parse containerd ports string into dict"""
        ports = {}
        if ports_str:
            parts = ports_str.split(', ')
            for part in parts:
                if '->' in part:
                    host, container = part.split('->')
                    ports[host.strip()] = container.strip()
        return ports


class ContainerRuntimeManager:
    """Manages multiple container runtimes and selects the best available one"""
    
    def __init__(self):
        self.runtimes = {
            'podman': PodmanRuntime(),
            'containerd': ContainerdRuntime(), 
            'docker': DockerRuntime()
        }
        self.preferred_runtime = None
        self.available_runtimes = []
    
    async def initialize(self, preferred_runtime: Optional[str] = None) -> None:
        """Initialize and detect available runtimes"""
        self.available_runtimes = []
        
        # Check which runtimes are available
        for name, runtime in self.runtimes.items():
            if await runtime.is_available():
                self.available_runtimes.append(name)
                logger.info(f"Detected available container runtime: {name}")
        
        if not self.available_runtimes:
            raise RuntimeError("No container runtime available. Please install Docker, Podman, or containerd.")
        
        # Set preferred runtime
        if preferred_runtime and preferred_runtime in self.available_runtimes:
            self.preferred_runtime = preferred_runtime
        else:
            # Default preference order: Podman > containerd > Docker
            preference_order = ['podman', 'containerd', 'docker']
            for runtime in preference_order:
                if runtime in self.available_runtimes:
                    self.preferred_runtime = runtime
                    break
        
        logger.info(f"Using container runtime: {self.preferred_runtime}")
    
    def get_runtime(self, runtime_name: Optional[str] = None) -> ContainerRuntime:
        """Get a specific runtime or the preferred one"""
        if runtime_name:
            if runtime_name not in self.available_runtimes:
                raise RuntimeError(f"Runtime {runtime_name} is not available")
            return self.runtimes[runtime_name]
        
        if not self.preferred_runtime:
            raise RuntimeError("No runtime initialized. Call initialize() first.")
        
        return self.runtimes[self.preferred_runtime]
    
    async def run_container(self, config: ContainerConfig, runtime_name: Optional[str] = None) -> str:
        """Run container using specified or preferred runtime"""
        runtime = self.get_runtime(runtime_name)
        return await runtime.run_container(config)
    
    async def auto_select_runtime_for_security(self) -> str:
        """Select the most secure available runtime"""
        # Preference for security: Podman (rootless) > containerd > Docker
        security_preference = ['podman', 'containerd', 'docker']
        
        for runtime in security_preference:
            if runtime in self.available_runtimes:
                logger.info(f"Selected {runtime} for enhanced security")
                return runtime
        
        raise RuntimeError("No secure container runtime available")
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get information about available runtimes"""
        return {
            'available_runtimes': self.available_runtimes,
            'preferred_runtime': self.preferred_runtime,
            'runtime_features': {
                'podman': {
                    'daemonless': True,
                    'rootless': True,
                    'docker_compatible': True,
                    'security_score': 9
                },
                'containerd': {
                    'daemonless': False,
                    'rootless': True,
                    'k8s_native': True,
                    'security_score': 8
                },
                'docker': {
                    'daemonless': False,
                    'rootless': False,
                    'widely_supported': True,
                    'security_score': 6
                }
            }
        }


# Global instance
_runtime_manager = None


async def get_runtime_manager() -> ContainerRuntimeManager:
    """Get the global runtime manager instance"""
    global _runtime_manager
    if _runtime_manager is None:
        _runtime_manager = ContainerRuntimeManager()
        await _runtime_manager.initialize()
    return _runtime_manager


async def create_secure_container(
    image: str,
    name: Optional[str] = None,
    command: Optional[Union[str, List[str]]] = None,
    environment: Optional[Dict[str, str]] = None,
    volumes: Optional[Dict[str, str]] = None,
    ports: Optional[Dict[str, str]] = None,
    prefer_secure_runtime: bool = True
) -> str:
    """Create a container using the most secure available runtime"""
    
    manager = await get_runtime_manager()
    
    # Use most secure runtime if preferred
    runtime_name = None
    if prefer_secure_runtime:
        runtime_name = await manager.auto_select_runtime_for_security()
    
    config = ContainerConfig(
        image=image,
        name=name,
        command=command,
        environment=environment,
        volumes=volumes,
        ports=ports,
        user="1000:1000" if runtime_name == 'podman' else None,  # Rootless for Podman
        security_opts=["no-new-privileges"] if runtime_name in ['podman', 'containerd'] else None
    )
    
    return await manager.run_container(config, runtime_name)