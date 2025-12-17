"""
Command-line interface for the 6G network simulation.

Provides entry points for running simulations and analysis.
"""

import argparse
import sys
from pathlib import Path

# Handle imports for both module execution and direct execution
try:
    # Try relative imports (works when run as module)
    from .simulation import run_simulation
    from .analysis import run_complete_analysis
except ImportError:
    try:
        # Try absolute imports (works when package is installed)
        from sixg_sim.simulation import run_simulation
        from sixg_sim.analysis import run_complete_analysis
    except ImportError:
        # Fallback: load modules directly (works when run as script)
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Import as if we're in the sixg_sim package
        import sixg_sim.simulation as simulation
        import sixg_sim.analysis as analysis
        run_simulation = simulation.run_simulation
        run_complete_analysis = analysis.run_complete_analysis


def main():
    """Main CLI entry point."""
    print("6G Network Simulation CLI starting...")

    parser = argparse.ArgumentParser(
        description="6G Network Simulation - Island Mode Proof of Concept"
    )

    parser.add_argument(
        "--topology", "-t",
        required=True,
        help="Path to topology configuration file (YAML/JSON)"
    )

    parser.add_argument(
        "--scenario", "-s",
        required=True,
        help="Path to scenario configuration file (YAML/JSON)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for results (default: output)"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation (default: 42)"
    )

    args = parser.parse_args()

    # Validate input files
    topology_path = Path(args.topology)
    scenario_path = Path(args.scenario)

    if not topology_path.exists():
        print(f"Error: Topology file not found: {topology_path}")
        sys.exit(1)

    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Arguments parsed successfully:")
        print(f"  topology: {args.topology}")
        print(f"  scenario: {args.scenario}")
        print(f"  output_dir: {args.output_dir}")

        if args.analyze_only:
            print("Analysis-only mode not yet implemented")
            # Would load existing metrics and run analysis
            sys.exit(1)
        else:
            # Run full simulation
            print("Starting 6G Network Simulation...")
            print(f"Topology: {topology_path}")
            print(f"Scenario: {scenario_path}")
            print(f"Output: {output_dir}")
            print()

            # Run simulation
            print("Running simulation...")
            from .simulation import run_simulation
            metrics = run_simulation(str(topology_path), str(scenario_path), str(output_dir))
            print("Simulation completed.")

            # Load topology for analysis (needed for node information)
            print("Loading topology for analysis...")
            try:
                from .topology import load_topology_from_yaml
            except ImportError:
                try:
                    from sixg_sim.topology import load_topology_from_yaml
                except ImportError:
                    import sixg_sim.topology as topology_module
                    load_topology_from_yaml = topology_module.load_topology_from_yaml
            topology = load_topology_from_yaml(str(topology_path))

            # Run analysis (already imported above)
            print("Running analysis...")
            run_complete_analysis(metrics, topology.nodes, str(output_dir))
            print(f"Analysis completed. Check output in {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
