import argparse
import asyncio
import base64
import bittensor as bt
import datetime
import hashlib
import os
import pandas as pd
import sys
import tempfile
import traceback

from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from dotenv import load_dotenv
from substrateinterface import SubstrateInterface
from typing import Any, Dict, Tuple, cast

from config.config_loader import load_config
from utils import (
    get_molecules,
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
)
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def parse_arguments() -> argparse.Namespace:
    """CONFIG & ARGUMENT PARSING"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", default=os.getenv("SUBTENSOR_NETWORK"), help="Network to use"
    )
    parser.add_argument("--netuid", type=int, default=68, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    config.update(load_config())
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            "miner",
        )
    )

    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """CONSTRUCT THE PATH FOR GITHUB OPERATION"""
    github_repo_name = os.environ.get("GITHUB_REPO_NAME")
    github_repo_branch = os.environ.get("GITHUB_REPO_BRANCH")
    github_repo_owner = os.environ.get("GITHUB_REPO_OWNER")
    github_repo_path = os.environ.get("GITHUB_REPO_PATH")

    if (
        github_repo_name is None
        or github_repo_branch is None
        or github_repo_owner is None
    ):
        raise ValueError(
            "Missing one or more GitHub environment variables (GITHUB_REPO_*)"
        )

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError(
            "GitHub path is too long. Please shorten it to 100 characters or less."
        )

    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    """SET UP BITTENSOR LOGGING"""
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:"
    )


async def setup_bittensor_objects(
    config: argparse.Namespace,
) -> Tuple[Any, Any, Any, int, int]:
    """BITTENSOR & NETWORK SETUP"""
    bt.logging.info("Setting up Bittensor objects.")

    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")

            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            node = SubstrateInterface(url=config.network)
            epoch_length = (
                node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
            )
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error(
            "Please check your network connection and the subtensor network status"
        )
        raise


async def run_psichic_model_loop(state: Dict[str, Any], config) -> None:
    """INFERENCE AND SUBMISSION LOGIC"""
    bt.logging.info("Starting PSICHIC model inference loop.")

    while not state["shutdown_event"].is_set():
        try:
            if state["shutdown_event"].is_set():
                break

            bt.logging.info("Selecting molecules from PyRMD package.")

            num_molecules_of_selection = 64
            num_thread_count = 32
            molecules = get_molecules(num_molecules_of_selection, num_thread_count)

            bt.logging.info(
                f"{num_molecules_of_selection} molecules was selected successfully."
            )

            product_names = []
            product_smiles = []
            for molecule in molecules:
                molecule_info = str(molecule).split("|")
                product_names.append(molecule_info[0])
                product_smiles.append(molecule_info[1])

            bt.logging.info(f"Selected product names: {product_names}")

            chunk = {
                "product_name": product_names,
                "product_smiles": product_smiles,
            }

            bt.logging.info("Data frame was generated successfully from chunk.")

            df = pd.DataFrame.from_dict(chunk)

            if df.empty or len(df) < state["config"].num_molecules:
                continue

            target_scores = []
            antitarget_scores = []

            bt.logging.info(
                "Inference task of target protein was started with selected molecules"
            )

            for target_protein in state["current_challenge_targets"]:
                if target_protein not in state["psichic_models"]:
                    try:
                        target_sequence = get_sequence_from_protein_code(target_protein)
                        model = PsichicWrapper()
                        model.run_challenge_start(target_sequence)
                        state["psichic_models"][target_protein] = model
                        bt.logging.info(
                            f"Initialized model for target: {target_protein}"
                        )
                    except Exception as e:
                        bt.logging.error(
                            f"Error initializing model for target {target_protein}: {e}"
                        )
                        continue

                scores = state["psichic_models"][target_protein].run_validation(
                    df["product_smiles"].tolist()
                )
                target_scores.append(scores[state["psichic_result_column_name"]])

            bt.logging.info(
                "Inference task of anti-target protein was started with selected molecules"
            )

            for antitarget_protein in state["current_challenge_antitargets"]:
                if antitarget_protein not in state["psichic_models"]:
                    try:
                        antitarget_sequence = get_sequence_from_protein_code(
                            antitarget_protein
                        )
                        model = PsichicWrapper()
                        model.run_challenge_start(antitarget_sequence)
                        state["psichic_models"][antitarget_protein] = model
                        bt.logging.info(
                            f"Initialized model for antitarget: {antitarget_protein}"
                        )
                    except Exception as e:
                        bt.logging.error(
                            f"Error initializing model for antitarget {antitarget_protein}: {e}"
                        )
                        continue

                scores = state["psichic_models"][antitarget_protein].run_validation(
                    df["product_smiles"].tolist()
                )
                antitarget_scores.append(scores[state["psichic_result_column_name"]])

            df["target_affinity"] = pd.DataFrame(target_scores).mean(axis=0)
            df["antitarget_affinity"] = pd.DataFrame(antitarget_scores).mean(axis=0)
            df["combined_score"] = (
                df["target_affinity"]
                - state["config"].antitarget_weight * df["antitarget_affinity"]
            )

            df.sort_values(by=["combined_score"], ascending=[False], inplace=True)
            df.reset_index(drop=True, inplace=True)

            top_molecules = df.iloc[:10]

            bt.logging.info(f"Top 10 molecules: {top_molecules}")

            if not top_molecules.empty:
                entropy = compute_maccs_entropy(
                    top_molecules["product_smiles"].tolist()
                )
                scores_sum = top_molecules["combined_score"].sum()

                if scores_sum > state["config"].entropy_bonus_threshold:
                    final_score = scores_sum * (
                        state["config"].entropy_start_weight + entropy
                    )
                else:
                    final_score = scores_sum

                bt.logging.info(f"Last candidate product: {state["candidate_product"]}")

                if final_score > state["best_score"]:
                    state["best_score"] = final_score
                    state["candidate_product"] = ",".join(
                        top_molecules["product_name"].tolist()
                    )
                    bt.logging.info(
                        f"New best score: {state['best_score']}, Candidates: {state['candidate_product']}"
                    )

                bt.logging.info(f"New candidate product: {state["candidate_product"]}")

                wallet, subtensor, metagraph, miner_uid, epoch_length = (
                    await setup_bittensor_objects(config)
                )
                current_block = await subtensor.get_current_block()
                next_epoch_block = (
                    (current_block // state["epoch_length"]) + 1
                ) * state["epoch_length"]
                blocks_until_epoch = next_epoch_block - current_block

                bt.logging.info(
                    f"Current block: {current_block}, Epoch length: {state['epoch_length']}, Next epoch block: {next_epoch_block}, Blocks until epoch: {blocks_until_epoch}"
                )

                if state["candidate_product"] and blocks_until_epoch <= 20:
                    bt.logging.info(
                        f"Close to epoch end ({blocks_until_epoch} blocks remaining), attempting submission..."
                    )
                    if state["candidate_product"] != state["last_submitted_product"]:
                        bt.logging.info("Attempting to submit new candidate...")
                        try:
                            await submit_response(state, config)
                        except Exception as e:
                            bt.logging.error(f"Error submitting response: {e}")
                    else:
                        bt.logging.info(
                            "Skipping submission - same product as last submission"
                        )
            else:
                bt.logging.info("Top molecules are empty.")

            await asyncio.sleep(2)

        except Exception as e:
            bt.logging.error(f"Error in PSICHIC model loop: {e}")
            traceback.print_exc()
            state["shutdown_event"].set()


async def submit_response(state: Dict[str, Any], config) -> None:
    """SUBMIT RESPONSE"""
    candidate_product = state["candidate_product"]
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")

    wallet, subtensor, metagraph, miner_uid, epoch_length = (
        await setup_bittensor_objects(config)
    )

    current_block = await subtensor.get_current_block()
    encrypted_response = state["bdt"].encrypt(
        state["miner_uid"], candidate_product, current_block
    )
    bt.logging.info(f"Encrypted response generated successfully")

    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, "w+") as f:
        f.write(str(encrypted_response))
        f.flush()
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

        bt.logging.info(f"Attempting chain commitment...")
        try:
            commitment_status = await subtensor.set_commitment(
                wallet=state["wallet"],
                netuid=state["config"].netuid,
                data=commit_content,
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info(
                "Too soon to commit again. Will keep looking for better candidates."
            )
            return

        if commitment_status:
            try:
                bt.logging.info(f"Commitment set successfully for {commit_content}")
                bt.logging.info("Attempting GitHub upload...")
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"File uploaded successfully to {commit_content}")
                    state["last_submitted_product"] = candidate_product
                    state["last_submission_time"] = datetime.datetime.now()
                else:
                    bt.logging.error(
                        f"Failed to upload file to GitHub for {commit_content}"
                    )
            except Exception as e:
                bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


async def run_miner(config: argparse.Namespace) -> None:
    """MAIN MINING LOOP"""
    wallet, subtensor, metagraph, miner_uid, epoch_length = (
        await setup_bittensor_objects(config)
    )

    state: Dict[str, Any] = {
        "config": config,
        "hugging_face_dataset_repo": "Metanova/SAVI-2020",
        "psichic_result_column_name": "predicted_binding_affinity",
        "chunk_size": 128,
        "submission_interval": 1200,
        "github_path": load_github_path(),
        "wallet": wallet,
        "subtensor": subtensor,
        "metagraph": metagraph,
        "miner_uid": miner_uid,
        "epoch_length": epoch_length,
        "psichic_models": {},
        "bdt": QuicknetBittensorDrandTimelock(),
        "candidate_product": None,
        "best_score": float("-inf"),
        "last_submitted_product": None,
        "last_submission_time": None,
        "shutdown_event": asyncio.Event(),
        "current_challenge_targets": [],
        "last_challenge_targets": [],
        "current_challenge_antitargets": [],
        "last_challenge_antitargets": [],
    }

    bt.logging.info("Entering main miner loop...")

    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12 * 10)
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets,
    )

    if startup_proteins:
        state["current_challenge_targets"] = startup_proteins["targets"]
        state["last_challenge_targets"] = startup_proteins["targets"]
        state["current_challenge_antitargets"] = startup_proteins["antitargets"]
        state["last_challenge_antitargets"] = startup_proteins["antitargets"]
        bt.logging.info(
            f"Startup targets: {startup_proteins['targets']}, antitargets: {startup_proteins['antitargets']}"
        )

        try:
            for target_protein in startup_proteins["targets"]:
                target_sequence = get_sequence_from_protein_code(target_protein)
                model = PsichicWrapper()
                model.run_challenge_start(target_sequence)
                state["psichic_models"][target_protein] = model
                bt.logging.info(f"Initialized model for target: {target_protein}")

            for antitarget_protein in startup_proteins["antitargets"]:
                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                model = PsichicWrapper()
                model.run_challenge_start(antitarget_sequence)
                state["psichic_models"][antitarget_protein] = model
                bt.logging.info(
                    f"Initialized model for antitarget: {antitarget_protein}"
                )
        except Exception as e:
            try:
                os.system(
                    f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')} "
                    f"https://huggingface.co/Metanova/TREAT-2/resolve/main/model.pt"
                )
                # Retry initialization after download
                for target_protein in state["current_challenge_targets"]:
                    if target_protein not in state["psichic_models"]:
                        target_sequence = get_sequence_from_protein_code(target_protein)
                        model = PsichicWrapper()
                        model.run_challenge_start(target_sequence)
                        state["psichic_models"][target_protein] = model
                        bt.logging.info(
                            f"Initialized model for target: {target_protein}"
                        )

                for antitarget_protein in state["current_challenge_antitargets"]:
                    if antitarget_protein not in state["psichic_models"]:
                        antitarget_sequence = get_sequence_from_protein_code(
                            antitarget_protein
                        )
                        model = PsichicWrapper()
                        model.run_challenge_start(antitarget_sequence)
                        state["psichic_models"][antitarget_protein] = model
                        bt.logging.info(
                            f"Initialized model for antitarget: {antitarget_protein}"
                        )
                bt.logging.info("Models re-downloaded and initialized successfully.")
            except Exception as e2:
                bt.logging.error(
                    f"Error initializing models after re-download attempt: {e2}"
                )

        try:
            state["inference_task"] = asyncio.create_task(
                run_psichic_model_loop(state, config)
            )
            bt.logging.info("Inference started on startup proteins.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")

    while True:
        bt.logging.info("Entering main miner loop...")
        try:
            current_block = await subtensor.get_current_block()

            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")

                block_hash = await subtensor.determine_block_hash(current_block)

                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets,
                )
                if new_proteins and (
                    new_proteins["targets"] != state["last_challenge_targets"]
                    or new_proteins["antitargets"]
                    != state["last_challenge_antitargets"]
                ):
                    state["current_challenge_targets"] = new_proteins["targets"]
                    state["last_challenge_targets"] = new_proteins["targets"]
                    state["current_challenge_antitargets"] = new_proteins["antitargets"]
                    state["last_challenge_antitargets"] = new_proteins["antitargets"]
                    bt.logging.info(
                        f"New proteins - targets: {new_proteins['targets']}, antitargets: {new_proteins['antitargets']}"
                    )

                if "inference_task" in state and state["inference_task"]:
                    if not state["inference_task"].done():
                        state["shutdown_event"].set()
                        bt.logging.info("Shutdown event set for old inference task.")
                        await state["inference_task"]

                state["candidate_product"] = None
                state["best_score"] = float("-inf")
                state["last_submitted_product"] = None
                state["shutdown_event"] = asyncio.Event()

                try:
                    for target_protein in state["current_challenge_targets"]:
                        if target_protein not in state["psichic_models"]:
                            target_sequence = get_sequence_from_protein_code(
                                target_protein
                            )
                            model = PsichicWrapper()
                            model.run_challenge_start(target_sequence)
                            state["psichic_models"][target_protein] = model
                            bt.logging.info(
                                f"Initialized model for target: {target_protein}"
                            )

                    for antitarget_protein in state["current_challenge_antitargets"]:
                        if antitarget_protein not in state["psichic_models"]:
                            antitarget_sequence = get_sequence_from_protein_code(
                                antitarget_protein
                            )
                            model = PsichicWrapper()
                            model.run_challenge_start(antitarget_sequence)
                            state["psichic_models"][antitarget_protein] = model
                            bt.logging.info(
                                f"Initialized model for antitarget: {antitarget_protein}"
                            )
                except Exception as e:
                    try:
                        os.system(
                            f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')} "
                            f"https://huggingface.co/Metanova/TREAT-2/resolve/main/model.pt"
                        )
                        # Retry initialization after download
                        for target_protein in state["current_challenge_targets"]:
                            if target_protein not in state["psichic_models"]:
                                target_sequence = get_sequence_from_protein_code(
                                    target_protein
                                )
                                model = PsichicWrapper()
                                model.run_challenge_start(target_sequence)
                                state["psichic_models"][target_protein] = model
                                bt.logging.info(
                                    f"Initialized model for target: {target_protein}"
                                )

                        for antitarget_protein in state[
                            "current_challenge_antitargets"
                        ]:
                            if antitarget_protein not in state["psichic_models"]:
                                antitarget_sequence = get_sequence_from_protein_code(
                                    antitarget_protein
                                )
                                model = PsichicWrapper()
                                model.run_challenge_start(antitarget_sequence)
                                state["psichic_models"][antitarget_protein] = model
                                bt.logging.info(
                                    f"Initialized model for antitarget: {antitarget_protein}"
                                )
                        bt.logging.info(
                            "Models re-downloaded and initialized successfully."
                        )
                    except Exception as e2:
                        bt.logging.error(
                            f"Error initializing models after re-download attempt: {e2}"
                        )

                try:
                    state["inference_task"] = asyncio.create_task(
                        run_psichic_model_loop(state, config)
                    )
                    bt.logging.info("New inference task started.")
                except Exception as e:
                    bt.logging.error(f"Error starting new inference: {e}")

            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break


async def main() -> None:
    """MAIN ENTRY POINT"""
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
