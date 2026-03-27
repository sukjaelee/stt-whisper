# stt-whisper

`stt-whisper` is a `faster-whisper`-based transcription workflow for Korean lecture-style audio. It is organized around a Google Colab notebook for interactive use, with an optional GCP Spot VM path for batch-style runs.

The project is tuned primarily for single-speaker recordings such as lectures and sermons. The recommended flow is to improve raw transcription quality first, review the raw transcript outputs, and enable LLM-based correction only after the base transcript is already acceptable.

Although the defaults target Korean audio, the same workflow can be adapted to other languages by changing `WHISPER_LANGUAGE` in the notebook config cell or CLI settings.

This project is shared openly in the hope that it may be useful to someone working on lecture and sermon transcription problems.

## Korean Summary

이 저장소는 한국어 강의형 음성을 전사하기 위한 워크플로를 제공합니다. 기본 경로는 Google Colab 노트북이고, 반복 실행이나 배치 처리를 위해 GCP Spot VM 작업 경로도 함께 제공합니다.
목표는 `faster-whisper`의 원시 출력 품질을 가능한 한 먼저 끌어올리고, 필요할 때만 후속 교정 단계를 적용하는 것입니다.

Colab은 설정을 바꿔가며 결과를 확인하기 쉬운 기본 실행 환경입니다. GCP는 Spot GPU VM을 이용해 파일 단위 작업을 반복 실행하거나 배치 처리하기 좋은 경로입니다. 전사 결과로 생성된 `.txt` 파일은 이후 LLM 기반 교정과 후처리에 활용할 수 있습니다.

실무에서는 보통 `SKIP_CORRECTION = True`로 먼저 원본 전사 결과를 확인한 뒤, 원 전사 품질이 충분히 괜찮을 때만 교정 단계를 켜는 것이 좋습니다.
현재 설정은 주로 한국어 강의형 음성에 맞춰져 있지만, 노트북이나 CLI에서 `WHISPER_LANGUAGE`를 바꾸면 다른 언어에도 같은 흐름을 적용할 수 있습니다.

강의, 설교 녹취 전사 문제를 다루는 누군가에게 이 프로젝트가 도움이 되길 바라며 공개합니다.

## Core Approach

- Model: `large-v3`
- Language: fixed to `ko` by default
- Runtime: prefer Colab GPU, fall back to CPU if needed
- Preprocessing: `ffmpeg` mono 16k WAV + high-pass + low-pass + loudness normalization
- Decoding: larger beam, tuned VAD, and less dependence on previous text
- Repetition control: temperature fallback, prompt reset, repetition penalty
- Outputs: both plain transcript and timestamped segment transcript

## Main Notebook

- Notebook: [notebooks/transcribe_to_txt_colab.ipynb](notebooks/transcribe_to_txt_colab.ipynb)

The Colab notebook is the main interactive entry point in this repository.

## Best Fit

This workflow is best suited to lecture-style, sermon-style, and other mostly single-speaker recordings.

It is less suitable for highly interactive conversations, interviews, meetings, or recordings with frequent speaker turns.

For conversational or multi-speaker audio, consider using a diarization-capable transcription model such as `gpt-4o-transcribe-diarize`. If you want higher-quality transcription without built-in diarization, `gpt-4o-transcribe` is another strong option.

## Colab Workflow

1. Put the repository in Google Drive.
   - Recommended folder layout:
     - `MyDrive/stt-whisper/`
       - `data/` (input/output audio + text files)
       - `notebooks/` (Colab notebook)
       - `prompts/` (LLM correction prompts)
       - `src/` (scripts: `transcribe_to_txt.py`, `gcp_job_runner.py` etc.)

2. Open [notebooks/transcribe_to_txt_colab.ipynb](notebooks/transcribe_to_txt_colab.ipynb) in Colab.

3. Run the install cell first.
   It installs `faster-whisper`, `openai`, `python-dotenv`, and `ffmpeg`.

4. Review the input file and transcription options in the config cell.
   `TRANSCRIBE_PRESET` can be set to one of:
   `high_quality`, `noise_robust`, `low_hallucination`

5. Start with `SKIP_CORRECTION = True` and inspect the raw transcript first.

6. Enable correction only after raw transcript quality is acceptable.

> NOTE: generated output filenames are described in **Generated Files** below.

## Generated Files

- `data/<input>_stt_ready.wav`: preprocessed audio for transcription
- `data/<input>.txt`: merged raw transcript
- `data/<input>_segments.txt`: timestamped segment transcript
- `data/<input>_direct_corrected.txt`: corrected transcript output

## Correction Step

The correction step runs after transcription and rewrites the raw transcript with an LLM using the selected prompt file.

Recommended use:

- First review the raw outputs: `.txt` and `_segments.txt`
- Then enable correction only if the base transcript is already reasonably accurate
- Use correction to improve readability, punctuation, spacing, and terminology consistency

Correction is not a replacement for a strong transcription model. If the raw transcript is substantially wrong, correction quality will also be limited.

## Prompt Files

Prompt files are stored in [prompts/](prompts) and are used only during the correction step.

Each file defines how aggressively the LLM should rewrite the transcript and which kinds of domain terms it should handle carefully.

Available prompt files:

- [prompts/direct_correction_default.md](prompts/direct_correction_default.md): balanced default correction
- [prompts/direct_correction_strict.md](prompts/direct_correction_strict.md): minimal correction with a conservative style
- [prompts/direct_correction_near_verbatim.md](prompts/direct_correction_near_verbatim.md): keeps the raw wording as close as possible
- [prompts/direct_correction_buddhist_terms.md](prompts/direct_correction_buddhist_terms.md): tuned for Buddhist terms and scripture references
- [prompts/direct_correction_sermon_cross_texts.md](prompts/direct_correction_sermon_cross_texts.md): tuned for mixed Buddhist, Biblical, and East Asian philosophy references

These prompt files are currently written in Korean because the primary correction workflow is aimed at Korean transcripts.

## Codex-Assisted Correction Without `OPENAI_API_KEY`

If you want to keep `SKIP_CORRECTION = True` in the notebook and avoid storing or using `OPENAI_API_KEY` in this project, you can use Codex as a manual or semi-automated correction step.

### Recommended Pattern

1. Transcribe the audio first with `SKIP_CORRECTION = True`.
2. Review the generated raw transcript in `data/<input>.txt` and, if needed, `data/<input>_segments.txt`.
3. In Codex, provide:
   - the raw transcript file
   - the prompt file to use from [prompts/](prompts)
   - any extra correction constraints for that file
4. Let Codex rewrite or clean up the transcript according to that prompt.
5. Save the output as a separate file such as `data/<input>_codex_corrected.txt`.

This pattern avoids project-level API key management, but it is not a fully unattended batch correction pipeline. A human still needs to invoke Codex for each correction run.

## Basic Automation Pattern

### API-Based Automation

For the transcription stage itself, the basic automation pattern can be implemented through APIs rather than by clicking through web UIs. In a typical implementation, file upload/download is handled through Google Cloud Storage APIs or Google Drive APIs, and VM lifecycle control is handled through the Compute Engine API.

### Workflow

1. Put `.mp3` files into `data/`.
2. Upload each file to cloud storage or another shared input location.
3. Start a GPU worker such as a GCP Spot T4 VM only when there is pending work.
4. Run transcription with `SKIP_CORRECTION = True`.
5. Upload the generated `.txt` and `_segments.txt` outputs.
6. Stop or delete the VM immediately after the job finishes.
7. Bring the raw outputs back into the local `data/` folder.
8. Use Codex with one of the prompt files for the correction pass if needed.

In practice, the cleanest cost-saving pattern is usually `submit job -> start Spot VM -> transcribe one file -> upload results -> delete VM`.

### Relevant API References

- Google Cloud Storage API: <https://cloud.google.com/storage/docs/apis>
- Google Cloud Storage upload objects: <https://cloud.google.com/storage/docs/uploading-objects>
- Google Cloud Storage download objects: <https://cloud.google.com/storage/docs/downloading-objects>
- Google Drive API: <https://developers.google.com/drive/api/guides/about-sdk>
- Google Drive API upload file data: <https://developers.google.com/workspace/drive/api/guides/manage-uploads>
- Google Drive API download files: <https://developers.google.com/workspace/drive/api/guides/manage-downloads>
- Compute Engine API: <https://cloud.google.com/compute/docs/reference/rest/v1>
- Compute Engine start and stop instances: <https://cloud.google.com/compute/docs/instances/stop-start-instance>
- Compute Engine create an N1 GPU VM: <https://cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose>

## GCS Automation Architecture

This is the recommended minimum architecture if you want to move from the Colab workflow to API-based automation on GCP.

### Required GCP Resources

- 1 GCP project for all resources
- 1 input bucket, for example `gs://<your-input-bucket>`
- 1 output bucket, for example `gs://<your-output-bucket>`
- 1 Spot GPU VM definition based on T4
- 1 service account attached to the VM
- 1 local launcher script that uploads jobs and creates VMs through APIs

### Recommended Data Flow

1. Put `data/<input>.mp3` in the local project.
2. Upload it to the input bucket.
3. Launch a Spot T4 VM for that file.
4. Pass the input object path and output prefix through instance metadata.
5. Let the VM startup script download the file, run transcription, and upload the results.
6. Delete the VM after the job finishes.
7. Download `data/<input>.txt` and `data/<input>_segments.txt` back to the local project.
8. If needed, run Codex-assisted correction on the raw `.txt`.

### IAM Roles

For the VM service account:

- `roles/storage.objectViewer` on the input bucket
- `roles/storage.objectCreator` or `roles/storage.objectAdmin` on the output bucket
- `roles/logging.logWriter` if you want Cloud Logging

For the launcher identity that creates the VM:

- `roles/compute.instanceAdmin.v1`
- `roles/iam.serviceAccountUser` on the VM service account
- `roles/storage.objectAdmin` on the buckets if the same launcher also uploads and downloads files

### Launcher Responsibilities

The launcher script should do the following:

1. Upload a local `.mp3` file to the input bucket.
2. Create a unique job id such as `<stem>-<timestamp>`.
3. Create a Spot T4 VM with:
   - the VM service account
   - a startup script
   - metadata for `INPUT_URI`, `OUTPUT_PREFIX`, and `JOB_ID`
   - a max runtime limit such as 30 minutes
4. Optionally poll for the expected output files.
5. Download the output files into the local `data/` folder.
6. Clear the `jobs/<job_id>/` objects from the input, staging, and output buckets after the waited job finishes.

Current repository entrypoint:

- [gcp_submit_job.py](gcp_submit_job.py): local launcher that uploads the input, uploads a source bundle, creates the Spot VM, optionally waits for completion, and downloads outputs

Local dependency note:

- install GCP launcher dependencies with `uv sync --extra gcp`

### VM Startup Script Responsibilities

The startup script should do the following:

1. Read `INPUT_URI`, `OUTPUT_PREFIX`, `JOB_ID`, and run options such as `TRANSCRIBE_PRESET` and `SKIP_CORRECTION` from instance metadata.
2. Install or activate the project runtime.
3. Download the input `.mp3` from GCS.
4. Run the transcription path with the configured `SKIP_CORRECTION` setting.
5. Upload:
   - `<stem>.txt`
   - `<stem>_segments.txt`
   - optionally `<stem>_stt_ready.wav`
6. Write a simple success or failure marker.
7. Shut down or delete the VM.

Current repository entrypoint:

- [gcp_startup_transcribe.sh](gcp_startup_transcribe.sh): startup script passed to the VM through instance metadata
- [src/gcp_job_runner.py](src/gcp_job_runner.py): VM-side runner that downloads from GCS, runs transcription, and uploads outputs
- [src/transcribe_to_txt.py](src/transcribe_to_txt.py): reusable CLI for the notebook-equivalent transcription path

### Why This MVP Is Preferred

- It avoids Google Drive ownership and shared-drive complexity.
- It works well with service accounts and unattended execution.
- It keeps cost low because the VM exists only for active jobs.
- It is easy to extend later to `GCS -> Pub/Sub -> Cloud Run -> Compute Engine`.

## GCP Setup

This section uses the following example values:

- project id: `<your-project-id>`
- zone: `<your-zone>`
- input bucket: `<your-input-bucket>`
- output bucket: `<your-output-bucket>`
- VM service account: `<your-vm-service-account>`

Note:

- The example above can use the default Compute Engine service account format for an MVP, but a dedicated VM service account is usually a better long-term choice.
- Cloud Storage bucket names must be globally unique. If your chosen input or output bucket name is already taken, create a different name and update the launcher command accordingly.

### Install `gcloud`

Windows:

1. Install the Google Cloud CLI by following the official Windows installer guide.
2. The installer can install bundled Python automatically.
3. After installation, open a new terminal window and verify:

```bash
gcloud --version
```

macOS:

1. Install the Google Cloud CLI by following the official macOS guide.
2. Choose the archive that matches your machine:
   - Apple silicon: `google-cloud-cli-darwin-arm.tar.gz`
   - Intel: `google-cloud-cli-darwin-x86_64.tar.gz`
3. After installation, open a new terminal and verify:

```bash
gcloud --version
```

Official installation guides:

- Google Cloud CLI install overview: <https://cloud.google.com/sdk/docs/install>
- Google Cloud CLI quickstart: <https://cloud.google.com/sdk/docs/install-sdk>

Command note:

- This README uses the standard `gcloud` command form in examples.
- On Windows PowerShell, if `gcloud.ps1` is blocked by the local execution policy, run the equivalent `gcloud.cmd` command instead or use a shell where `gcloud` resolves normally.

### Authenticate Locally

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>
```

`gcp_submit_job.py` uses Application Default Credentials on the local machine, so `gcloud auth application-default login` is required.

References:

- Application Default Credentials overview: <https://cloud.google.com/docs/authentication/provide-credentials-adc>
- `gcloud auth application-default login`: <https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login>

### Enable Required APIs

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com
```

### Create Buckets

```bash
gcloud storage buckets create gs://<your-input-bucket> --location=<your-region> --project=<your-project-id>
gcloud storage buckets create gs://<your-output-bucket> --location=<your-region> --project=<your-project-id>
```

Reference:

- Cloud Storage bucket creation: <https://cloud.google.com/storage/docs/creating-buckets>

### Grant Bucket Access To The VM Service Account

```bash
gcloud storage buckets add-iam-policy-binding gs://<your-input-bucket> --member="serviceAccount:<your-vm-service-account>" --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding gs://<your-output-bucket> --member="serviceAccount:<your-vm-service-account>" --role="roles/storage.objectAdmin"
```

### Grant Local Launcher Permissions

If the local launcher will be run by `<your-admin-email>`, grant:

```bash
gcloud projects add-iam-policy-binding <your-project-id> --member="user:<your-admin-email>" --role="roles/compute.instanceAdmin.v1"
gcloud iam service-accounts add-iam-policy-binding <your-vm-service-account> --member="user:<your-admin-email>" --role="roles/iam.serviceAccountUser"
```

### Install Local Python Dependencies

Python version note:

- This project now expects `Python 3.11+`.
- `faster-whisper` currently pulls `onnxruntime`, and the locked `onnxruntime` build used here does not provide wheels for `Python 3.10`.

If `uv sync --extra gcp` created a Python 3.10 virtual environment, remove it first and recreate it with Python 3.11 or newer:

```bash
rm -rf .venv
uv python install 3.11
uv sync --python 3.11 --extra gcp
```

### Save GCP Defaults In `.env`

If you want to avoid repeating the same GCP options on every run, create a `.env` file in the project root and fill in the GCP launcher values.

Key variables:

- `GCP_PROJECT_ID`
- `GCP_ZONE`
- `GCP_SERVICE_ACCOUNT`
- `GCP_INPUT_BUCKET`
- `GCP_OUTPUT_BUCKET`
- `GCP_IMAGE_FAMILY`
- `GCP_IMAGE_PROJECT`
- `GCP_WHISPER_MODEL_PATH`
- `GCP_OPENAI_API_KEY_SECRET`
- `GCP_SKIP_CORRECTION`
- `GCP_WAIT`
- `GCP_DOWNLOAD_DIR`

`GCP_SKIP_CORRECTION` defaults to `true`. If you set it to `false` or pass `--no-skip-correction`, make sure the VM can resolve `OPENAI_API_KEY`.

Recommended setup:

- store the key in Secret Manager
- grant the VM service account `roles/secretmanager.secretAccessor`
- set `GCP_OPENAI_API_KEY_SECRET` to a full resource such as `projects/<project>/secrets/openai-api-key/versions/latest`

At VM startup, the launcher passes that secret resource through instance metadata and the startup script exports the fetched value as `OPENAI_API_KEY` before running correction.

Example Secret Manager setup:

1. Create the secret once:

```bash
gcloud secrets create openai-api-key \
  --project <your-project-id> \
  --replication-policy="automatic"
```

2. Add the actual API key as a secret version:

```bash
printf '%s' '<your-openai-api-key>' | gcloud secrets versions add openai-api-key \
  --project <your-project-id> \
  --data-file=-
```

3. Grant the VM service account permission to read that secret:

```bash
gcloud secrets add-iam-policy-binding openai-api-key \
  --project <your-project-id> \
  --member="serviceAccount:<your-vm-service-account>" \
  --role="roles/secretmanager.secretAccessor"
```

4. Point `.env` at that secret resource:

```env
GCP_OPENAI_API_KEY_SECRET=projects/<your-project-id>/secrets/openai-api-key/versions/latest
GCP_SKIP_CORRECTION=false
```

`gcloud secrets add-iam-policy-binding` is the step that allows the VM service account to read the secret at startup.
If that binding is missing or is granted to the wrong service account or project, the VM will fail with a Secret Manager `403` when correction is enabled.

Once those values are set in `.env`, the launcher can usually be run with just the input file path:

```bash
uv run python gcp_submit_job.py data/sample_lecture.mp3
```

If you prefer a shorter entrypoint, use the root-level wrapper script:

```bash
./submit_gcp_job.sh data/sample_lecture.mp3
```

You can still pass through extra launcher arguments when needed:

```bash
./submit_gcp_job.sh data/sample_lecture.mp3 --no-clear-job-objects
```

`submit_gcp_job.sh` is a thin wrapper around `uv run python gcp_submit_job.py`.
It uses the first argument as the input audio path and forwards every remaining argument to `gcp_submit_job.py` unchanged.

That means the wrapper can pass through flags such as `--no-clear-job-objects` and `--no-skip-correction`.

Example:

```bash
./submit_gcp_job.sh data/sample_lecture.mp3 --no-skip-correction
```

If `.env` already contains `GCP_OPENAI_API_KEY_SECRET` and `GCP_SKIP_CORRECTION=false`, then `./submit_gcp_job.sh <input-audio-path>` will also run the correction step without any extra CLI flags.

Boot disk note:

- The launcher defaults to `--boot-disk-size-gb 60` for the base Deep Learning VM image family.
- When you use the `stt-whisper` custom image family in your own project, the launcher defaults to `--boot-disk-size-gb 80`.
- If you override it manually, keep it at `60` or higher for the base image family and `80` or higher for the current `stt-whisper` custom image family.

macOS or Linux shell form:

```bash
rm -rf .venv
uv python install 3.11
uv sync --python 3.11 --extra gcp
```

### Run A First End-To-End Test

```bash
uv run python gcp_submit_job.py data/sample_lecture.mp3 \
  --project-id <your-project-id> \
  --zone <your-zone> \
  --input-bucket <your-input-bucket> \
  --output-bucket <your-output-bucket> \
  --service-account <your-vm-service-account> \
  --wait \
  --download-dir data
```

Cleanup behavior:

- When `--wait` is used, the launcher now deletes the current job's `jobs/<job_id>/` objects from the input bucket, the staging bucket, and the output bucket after the result files are downloaded.
- If you want to keep the remote job artifacts for debugging, add `--no-clear-job-objects`.

### Use A Custom Image With A Local `large-v3` Cache

Spot VM instances can use both:

- a custom Compute Engine image
- a local baked Whisper model directory

The repository now supports passing a baked local model path to the VM with:

- `--whisper-model-path /opt/stt-whisper-models/large-v3`

The startup script will pass that path through to the Python runner, and if the directory exists it will use it instead of downloading `large-v3` from the Hugging Face cache path.

Recommended flow:

1. Create a one-time builder VM from the current Deep Learning VM base image.
2. Copy this repository to the builder VM, or at minimum upload [gcp_prepare_custom_image.sh](gcp_prepare_custom_image.sh).
3. Run [gcp_prepare_custom_image.sh](gcp_prepare_custom_image.sh) on the builder VM as root.
4. Create a custom image from that prepared builder disk.
5. Launch future Spot VMs from that custom image family and pass `--whisper-model-path`.

Example builder VM creation:

```bash
gcloud compute instances create stt-whisper-image-builder-v2 \
  --project=<your-project-id> \
  --zone=<your-zone> \
  --machine-type=n1-standard-4 \
  --boot-disk-size=80GB \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release
```

Then SSH into the builder VM, place this repository on the machine, and run:

```bash
sudo bash ./gcp_prepare_custom_image.sh
```

Script locations in this repository:

- [gcp_prepare_custom_image.sh](gcp_prepare_custom_image.sh): root-level helper for preparing the builder VM before image creation
- [gcp_startup_transcribe.sh](gcp_startup_transcribe.sh): root-level startup script passed to Spot VMs at job launch time
- [submit_gcp_job.sh](submit_gcp_job.sh): root-level wrapper around `uv run python gcp_submit_job.py`

That script installs the runtime packages and downloads the converted `large-v3` model into:

- `/opt/stt-whisper-models/large-v3`

Then create a reusable custom image:

```bash
gcloud compute images create stt-whisper-cu128-v2 \
  --project=<your-project-id> \
  --source-disk=stt-whisper-image-builder-v2 \
  --source-disk-zone=<your-zone> \
  --family=stt-whisper-cu128
```

After that, launch Spot jobs with the custom image family and the baked model path:

```bash
uv run python gcp_submit_job.py data/sample_lecture.mp3 \
  --project-id <your-project-id> \
  --zone <your-zone> \
  --input-bucket <your-input-bucket> \
  --output-bucket <your-output-bucket> \
  --service-account <your-vm-service-account> \
  --image-family stt-whisper-cu128 \
  --image-project <your-project-id> \
  --whisper-model-path /opt/stt-whisper-models/large-v3 \
  --wait \
  --download-dir data
```

This reduces cold-start time by avoiding repeated `pip install` work and by keeping the converted `large-v3` model on the VM image itself.
The launcher also defaults to an `80 GiB` boot disk for the `stt-whisper` custom image family, so you usually do not need to pass `--boot-disk-size-gb` explicitly in that mode.
The current custom image preparation flow may still create a baked virtualenv with the base image's default `python3`, but the local project environment should remain on Python `3.11+`.

### Operational Checks

- Confirm that your chosen zone has available T4 quota for your project.
- Confirm that the VM service account is allowed to access both buckets.
- If VM creation fails because of bucket names, change them to globally unique names and rerun the setup commands.
- If VM creation fails because of GPU availability, choose another T4-capable zone and update the launcher command.
- The first VM boot can take noticeably longer because the startup script installs the NVIDIA driver on Debian 12 before transcription starts.
- The base Deep Learning VM image path uses a default `60 GiB` boot disk.
- The current `stt-whisper` custom image path uses a default `80 GiB` boot disk because the custom image was built from an `80 GB` source disk.
- If you use a custom image prepared with [gcp_prepare_custom_image.sh](gcp_prepare_custom_image.sh), the startup script will skip most `apt` and `pip` bootstrap work and can start transcription noticeably faster.

Recommended image default:

- The launcher now defaults to the Deep Learning VM image family `common-cu128-ubuntu-2204-nvidia-570` from the `deeplearning-platform-release` project so that GPU-capable images start with NVIDIA support already present.

### Verify GPU Availability On The VM

After `gcp_submit_job.py` prints the created `vm_name`, confirm GPU access:

```bash
gcloud compute ssh VM_NAME --zone <your-zone> --command "nvidia-smi"
```

If `nvidia-smi` is not found, that VM is not ready for GPU transcription and `faster-whisper` will fall back to CPU.

### View Job Logs On The VM

The startup script now redirects its own output and the Python runner output to:

- `/var/log/stt-whisper.log`

To watch progress in real time:

```bash
gcloud compute ssh VM_NAME --zone <your-zone> --command "sudo tail -f /var/log/stt-whisper.log"
```

To read the latest lines once:

```bash
gcloud compute ssh VM_NAME --zone <your-zone> --command "sudo tail -n 200 /var/log/stt-whisper.log"
```

Typical stages you should see:

- startup script begin
- metadata loaded
- GPU status
- downloading source bundle
- launching Python job runner
- downloading input
- starting transcription
- runtime device: `cuda`
- uploading transcript outputs
- uploading success marker
- shutting down VM

If you need startup-script system logs in addition to the app log:

```bash
gcloud compute ssh VM_NAME --zone <your-zone> --command "sudo journalctl -u google-startup-scripts.service -n 200 --no-pager"
```

## Cost Example: GCP Spot T4

### Example Workload

- 30 files per month
- reference lecture length: about 1 hour 37 minutes
- reference prepared audio duration: `Estimated total duration: 01:30:09`
- observed processing time on the GPU worker: `Transcription elapsed time: 00:19:13`

### Monthly GPU Runtime

- `30 files x 19 minutes 13 seconds = 576 minutes 30 seconds = 9 hours 36 minutes 30 seconds`

### Cost Assumptions

- Spot T4 GPU: `$0.14 / hour`
- Spot host VM: `n1-standard-4` equivalent
- Spot host VM estimate:
  - vCPU: `4 x $0.00811 / hour`
  - memory: `15 GiB x $0.001039 / GiB-hour`
  - host total: about `$0.048 / hour`

### Estimated Monthly Processing Cost

- GPU only: `9.6083 x $0.14 = about $1.35 / month`
- host VM only: `9.6083 x $0.048 = about $0.46 / month`
- GPU + host VM: `about $1.81 / month`

### Disk Notes

- If you create and delete the VM per job and keep only a small `pd-standard` boot disk, disk cost can stay near zero or remain negligible.
- If you keep a `pd-balanced` boot disk around for the whole month, add roughly another `$1 to $3 / month` depending on disk size.

### Monthly Estimate Summary

So for this workload, a realistic GCP Spot T4 monthly estimate is:

- around `$1.81 / month` for pure processing time
- around `$2.81 to $4.81 / month` if you retain a small balanced boot disk during the month

These numbers assume the VM runs only while jobs are active. If you leave the VM running all month, the cost will be much higher.

## Default Quality Settings

These are the current defaults for transcription quality:

- `TRANSCRIBE_PRESET = "high_quality"`
- `WHISPER_MODEL_SIZE = "large-v3"`
- `WHISPER_LANGUAGE = "ko"`
- `WHISPER_BEAM_SIZE = 15`
- `WHISPER_BEST_OF = 5`
- `WHISPER_PATIENCE = 1.6`
- `WHISPER_TEMPERATURES = (0.0, 0.2, 0.4, 0.6)`
- `WHISPER_PROMPT_RESET_ON_TEMPERATURE = 0.5`
- `WHISPER_REPETITION_PENALTY = 1.05`
- `WHISPER_NO_REPEAT_NGRAM_SIZE = 3`
- `WHISPER_MULTILINGUAL = False`
- `WHISPER_VAD_MIN_SILENCE_MS = 500`
- `WHISPER_VAD_SPEECH_PAD_MS = 500`
- `WHISPER_ENABLE_DENOISE = True`

You can adjust these directly in the notebook config cell.

References:

- Official `faster-whisper` repository: <https://github.com/SYSTRAN/faster-whisper>
- `faster-whisper` README examples and VAD notes: <https://github.com/SYSTRAN/faster-whisper#readme>
- `WhisperModel` implementation for available transcription options: <https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py>

## Recommended Presets

- `high_quality`: prioritizes the best transcript quality and may be the slowest
- `noise_robust`: tuned for background noise, reverberation, and mild overlap
- `low_hallucination`: tuned more conservatively to reduce repetition and hallucination

## Code Structure

- [gcp_submit_job.py](gcp_submit_job.py): local GCP launcher for upload, VM creation, result polling, download, and cleanup
- [gcp_prepare_custom_image.sh](gcp_prepare_custom_image.sh): builder-VM helper for preparing a reusable custom image
- [gcp_startup_transcribe.sh](gcp_startup_transcribe.sh): VM startup script used for Spot transcription jobs
- [submit_gcp_job.sh](submit_gcp_job.sh): convenience wrapper for common GCP submission runs
- [src/gcp_job_runner.py](src/gcp_job_runner.py): VM-side orchestration for one transcription job
- [src/transcription_utils.py](src/transcription_utils.py): preprocessing, runtime fallback, transcript merging, and output writing
- [src/direct_correction.py](src/direct_correction.py): correction step
- [src/path_utils.py](src/path_utils.py): path resolution helpers
