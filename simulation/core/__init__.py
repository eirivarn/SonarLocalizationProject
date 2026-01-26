"""
Core simulation engine for SOLAQUA sonar simulation.

Exports:
- VoxelGrid, VoxelSonar: Base simulation classes
- VoxelGridWithMaterials, VoxelSonarWithSegmentation: Material tracking extensions
- Materials: NET, ROPE, FISH, BIOMASS, etc.
- Scene generation: create_random_scene, generate_random_sonar_position
- Coordinate conversion: polar_to_cartesian
- Config: SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG, etc.
"""
from .simulation import (
    VoxelGrid, VoxelSonar, Material,
    EMPTY, NET, ROPE, FISH, BIOMASS,
    DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY
)
from .config import (
    SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG,
    MATERIAL_CONFIG, DATASET_DIR
)
from .data_generator import (
    create_random_scene, 
    generate_random_sonar_position,
    polar_to_cartesian,
    generate_sample
)
from .semantic_segmentation import (
    VoxelGridWithMaterials,
    VoxelSonarWithSegmentation,
    MATERIAL_IDS,
    MATERIAL_COLORS,
    create_semantic_visualization
)

__all__ = [
    # Base classes
    'VoxelGrid', 'VoxelSonar', 'Material',
    # Materials
    'EMPTY', 'NET', 'ROPE', 'FISH', 'BIOMASS',
    'DEBRIS_LIGHT', 'DEBRIS_MEDIUM', 'DEBRIS_HEAVY',
    # Config
    'SONAR_CONFIG', 'SCENE_CONFIG', 'DATA_GEN_CONFIG',
    'MATERIAL_CONFIG', 'DATASET_DIR',
    # Scene generation
    'create_random_scene', 'generate_random_sonar_position',
    'polar_to_cartesian', 'generate_sample',
    # Semantic segmentation
    'VoxelGridWithMaterials', 'VoxelSonarWithSegmentation',
    'MATERIAL_IDS', 'MATERIAL_COLORS', 'create_semantic_visualization',
]
