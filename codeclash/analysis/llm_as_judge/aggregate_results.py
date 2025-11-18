#!/usr/bin/env python3

"""Aggregate results from multiple llm_as_judge.json files
and save as a compressed parquet file for efficient storage and loading.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from codeclash.analysis.llm_as_judge.categorize_actions import _all_categories as ACTION_CATEGORIES
from codeclash.analysis.llm_as_judge.hallucination import claim_categories as CLAIM_CATEGORIES
from codeclash.analysis.llm_as_judge.hallucination import source_categories as SOURCE_CATEGORIES
from codeclash.analysis.llm_as_judge.utils import Instance, InstanceBatch
from codeclash.utils.log import get_logger

logger = get_logger("AggregateResults", emoji="📊")

# Version constants for specific data_ids
BIG_QUESTIONS_VERSION = 7
ACTION_CATEGORIES_VERSION = 3
HALLUCINATION_VERSION = 17

# Generate all hallucination category combinations
HALLUCINATION_CATEGORIES = [f"{claim}__{source}" for claim in CLAIM_CATEGORIES for source in SOURCE_CATEGORIES]


def _read_judge_file(file_path: Path) -> dict | None:
    """Read and parse a single llm_as_judge.json file.

    Args:
        file_path: Path to the llm_as_judge.json file to read.

    Returns:
        Parsed JSON data (dict with data_id keys mapping to instance data) or None if file is empty/invalid.
    """
    try:
        content = file_path.read_text().strip()
        if not content:
            logger.warning(f"Skipping empty file: {file_path}")
            return None
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


class ResultsAggregator:
    """Class for aggregating LLM-as-judge evaluation results."""

    def __init__(self):
        self.big_questions_data_id = f"big_questions_v{BIG_QUESTIONS_VERSION}"
        self.action_categories_data_id = f"action_categories_v{ACTION_CATEGORIES_VERSION}"
        self.hallucination_data_id = f"hallucination_v{HALLUCINATION_VERSION}"

        # Track counts during aggregation
        self.action_categories_count = 0
        self.big_questions_count = 0
        self.hallucination_count = 0

    def aggregate_results_to_dataframe(self, input_dir: Path, *, instance_ids: set[str] | None = None) -> pd.DataFrame:
        """Aggregate all llm_as_judge.json results from the input directory into a DataFrame.

        Args:
            input_dir: Directory containing llm_as_judge.json files
            instance_ids: If provided, only include results for these instance IDs

        Returns:
            DataFrame with flattened structure containing all evaluation data merged by instance_id
        """
        # Dictionary to collect all data by instance_id
        instance_data_dict = {}
        llm_judge_files = list(input_dir.rglob("llm_as_judge.json"))

        logger.info(f"Found {len(llm_judge_files)} llm_as_judge.json files")
        if instance_ids is not None:
            logger.info(f"Filtering to {len(instance_ids)} specific instances")

        # Read files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(_read_judge_file, file_path): file_path for file_path in llm_judge_files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                file_data = future.result()

                if file_data is None:
                    continue

                logger.debug(f"Processing {file_path}")

                try:
                    # Process each data_id and instance
                    for data_id, instances in file_data.items():
                        # Only process allowed data_ids
                        if data_id not in [
                            self.action_categories_data_id,
                            self.big_questions_data_id,
                            self.hallucination_data_id,
                        ]:
                            continue

                        for instance_id, instance_data in instances.items():
                            # Skip if we're filtering and this instance isn't in the filter set
                            if instance_ids is not None and instance_id not in instance_ids:
                                continue

                            # Initialize instance data if not seen before
                            if instance_id not in instance_data_dict:
                                instance_data_dict[instance_id] = self._initialize_instance_row(instance_data)

                            # Add data specific to this data_id
                            self._add_data_id_results(instance_data_dict[instance_id], data_id, instance_data)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}", exc_info=True)

        # Check if all expected instances were found
        if instance_ids is not None:
            found_instance_ids = set(instance_data_dict.keys())
            missing_instance_ids = instance_ids - found_instance_ids
            if missing_instance_ids:
                raise ValueError(
                    f"Could not find {len(missing_instance_ids)} instances in llm_as_judge.json files: {sorted(missing_instance_ids)}"
                )

        # Convert to DataFrame directly from dict_values (no need for list conversion)
        df = pd.DataFrame(instance_data_dict.values())

        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Instances with {self.action_categories_data_id}: {self.action_categories_count}")
        logger.info(f"Instances with {self.big_questions_data_id}: {self.big_questions_count}")
        logger.info(f"Instances with {self.hallucination_data_id}: {self.hallucination_count}")

        sorted_columns = "\n".join(sorted(df.columns.tolist()))
        logger.info(f"DataFrame columns (sorted): {sorted_columns}")

        return df

    def _initialize_instance_row(self, instance_data: dict) -> dict:
        """Initialize a row with basic instance metadata only."""
        instance = Instance.model_validate(instance_data["instance"])
        model_name, opponent_model_name = instance.get_lm_name_self_opponent()
        current_round_win_rate, next_round_win_rate = instance.get_current_next_round_win_rate()

        # Create base row with instance metadata only
        # Categories will be initialized lazily when data_id results are added
        return {
            "instance_id": instance.instance_id,
            "tournament_name": instance.tournament_name,
            "player_name": instance.player_name,
            "round_number": instance.round_number,
            "model_name": model_name,
            "opponent_model_name": opponent_model_name,
            "current_round_win_rate": current_round_win_rate,
            "next_round_win_rate": next_round_win_rate,
        }

    def _add_data_id_results(self, row: dict, data_id: str, instance_data: dict) -> None:
        """Add results from a specific data_id to the row and track counts."""
        # Add the data_id information
        if "data_ids" not in row:
            row["data_ids"] = []

        # Track counts only on first occurrence for this instance
        if data_id not in row["data_ids"]:
            if data_id == self.action_categories_data_id:
                self.action_categories_count += 1
            elif data_id == self.big_questions_data_id:
                self.big_questions_count += 1
            elif data_id == self.hallucination_data_id:
                self.hallucination_count += 1

        row["data_ids"].append(data_id)

        # Delegate to specific handlers based on data_id type
        if data_id == self.action_categories_data_id:
            self._add_action_categories_results(row, instance_data)
        elif data_id == self.big_questions_data_id:
            self._add_big_questions_results(row, instance_data)
        elif data_id == self.hallucination_data_id:
            self._add_hallucination_results(row, instance_data)
        # Ignore other data_ids that are not handled

    def _add_action_categories_results(self, row: dict, instance_data: dict) -> None:
        """Add action categories counts to the row."""
        if "result" not in instance_data:
            return

        result_data = instance_data["result"]

        # Count occurrences of each category using Counter (more efficient)
        categories = result_data.get("categories", [])
        if categories:
            category_counts = Counter(cat.get("category") for cat in categories if cat.get("category"))
            # Only set counts for valid categories (filter invalid ones)
            for category in ACTION_CATEGORIES:
                row[f"c_{category}"] = category_counts.get(category, 0)
        else:
            # Initialize all to 0 if no categories found
            for category in ACTION_CATEGORIES:
                row[f"c_{category}"] = 0

    def _add_big_questions_results(self, row: dict, instance_data: dict) -> None:
        """Add big questions results to the row."""
        if "result" not in instance_data:
            return

        result_data = instance_data["result"]

        # Add all result data without prefix
        for key, value in result_data.items():
            row[key] = value

    def _add_hallucination_results(self, row: dict, instance_data: dict) -> None:
        """Add hallucination counts to the row."""
        if "result" not in instance_data:
            return

        result_data = instance_data["result"]

        # Build combinations and count them efficiently
        items = result_data.get("items", [])
        if items:
            # Use generator directly with Counter for better memory efficiency
            combination_counts = Counter(
                f"{item['claim_category']}__{item['source_category']}"
                for item in items
                if item.get("claim_category") and item.get("source_category")
            )

            # Set counts for all hallucination categories
            for category in HALLUCINATION_CATEGORIES:
                row[f"h_{category}"] = combination_counts.get(category, 0)
        else:
            # Initialize all to 0 if no items found
            for category in HALLUCINATION_CATEGORIES:
                row[f"h_{category}"] = 0


def aggregate_results_to_dataframe(input_dir: Path) -> pd.DataFrame:
    """Convenience function for backward compatibility."""
    aggregator = ResultsAggregator()
    return aggregator.aggregate_results_to_dataframe(input_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LLM-as-judge evaluation results to Parquet")
    parser.add_argument("input_dir", type=Path, help="Path to the input directory containing tournament results")
    parser.add_argument(
        "-o", "--output-file", type=Path, help="Path to the output Parquet file", default="aggregated_results.parquet"
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        help="Path to instances.json file (output of get_instances.py) to filter to specific instances",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    # Load instance filter if provided
    instance_ids = None
    if args.instance_file is not None:
        if not args.instance_file.exists():
            raise FileNotFoundError(f"Instance file does not exist: {args.instance_file}")
        instance_batch = InstanceBatch.model_validate_json(args.instance_file.read_text())
        instance_ids = {instance.instance_id for instance in instance_batch.instances}
        logger.info(f"Loaded {len(instance_ids)} instances from {args.instance_file}")

    logger.info(f"Aggregating results from {args.input_dir}")
    aggregator = ResultsAggregator()
    df = aggregator.aggregate_results_to_dataframe(args.input_dir, instance_ids=instance_ids)

    df.to_parquet(args.output_file, compression="snappy", index=False)
    logger.info(f"Wrote aggregated results to {args.output_file}")


if __name__ == "__main__":
    main()
