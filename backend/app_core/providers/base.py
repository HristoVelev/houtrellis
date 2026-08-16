from abc import ABC, abstractmethod
from typing import Any, Dict


class Base2DProvider(ABC):
    """Abstract Base Class Strategy interface for all 2D reference image providers."""

    @abstractmethod
    def load_model(self) -> Any:
        """Loads and returns the generative pipeline onto GPU."""
        pass

    @abstractmethod
    def generate_2d(
        self, prompt: str, seed: int, params: Dict[str, Any], output_path: str
    ) -> str:
        """
        Generates a 2D reference PNG image and saves it to output_path.
        Returns the absolute path to the generated PNG file.
        """
        pass


class Base3DProvider(ABC):
    """Abstract Base Class Strategy interface for all 3D mesh generative providers."""

    @abstractmethod
    def load_model(self) -> Any:
        """Loads and returns the generative pipeline onto GPU."""
        pass

    @abstractmethod
    def generate_3d(
        self, image_path: str, seed: int, params: Dict[str, Any], output_path: str
    ) -> str:
        """
        Generates a 3D GLB mesh asset from image_path and saves it to output_path.
        Returns the absolute path to the generated GLB file.
        """
        pass
