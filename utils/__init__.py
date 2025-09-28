# Re-export main entry points for convenience.
# Import from available modules only

try:
    from .reporting import (
        load_data_index, list_exported_bag_stems, overview_by_bag, overview_by_datetime,
        topics_in_bag_df, topics_overview_dir,
    )
except ImportError:
    pass

try:
    # Import from sonar_utils as needed by other modules
    from .sonar_utils import (
        ConeGridSpec, load_df, enhance_intensity, iter_cone_frames, 
        save_cone_run_npz, get_sonoptix_frame, parse_json_cell
    )
except ImportError:
    pass

try:
    # Import from new sonar visualization module
    from .sonar_visualization import (
        SonarVisualizer, find_sonar_files, print_sonar_files, quick_visualize
    )
except ImportError:
    pass

try:
    from .image_analysis_utils import *
except ImportError:
    pass

try:
    from .navigation_guidance_analysis import *
except ImportError:
    pass

try:
    from .net_line_utils import *
except ImportError:
    pass

try:
    from .nucleus1000dvl_analysis import *
except ImportError:
    pass

try:
    from .sonar_distance_analysis import *
except ImportError:
    pass

try:
    from .synchronized_analysis import *
except ImportError:
    pass

# Optional zero-shot model utilities (requires additional dependencies)
try:
    from .zero_shot_utils import (
        analyze_cone_run_with_zero_shot, clip_score_cone_objects,
        check_model_availability, get_installation_commands
    )
except ImportError:
    # Zero-shot utilities unavailable (missing transformers/torch/segment-anything)
    pass
