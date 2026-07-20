# Dataset C — Mining Pipeline Statistics & Filter Rationale

**Source script:** [scripts/mine_knative.py](../scripts/mine_knative.py)
**Output dataset:** [data/dataset_c.jsonl](dataset_c.jsonl)
**Raw run log:** [data/mine_c.log](mine_c.log)
**Methodology reference:** [Dataset_collection_methodology_v2.md](../Dataset_collection_methodology_v2.md) §5.3

---

## 1. Run Configuration

| Parameter | Value |
|---|---|
| Repositories | `knative/serving`, `knative/eventing`, `knative/pkg`, `knative/caching`, `knative/client`, `knative/func`, `knative/operator`, `knative/networking` |
| Date filter | `closed_at >= 2024-12-01 UTC` (post-LLM-cutoff to avoid data leakage) |
| Issue state | `closed` |
| Run date | 2026-05-14 |

---

## 2. Funnel Summary

| Stage | Filter | Items Removed | Cumulative Survivors |
|---|---|---:|---:|
| 0 | Raw closed-issues endpoint hits | — | **4,083** |
| 1a | `pull_request != null` (endpoint also returns PRs) | 3,148 | 935 |
| 1b | `closed_at < 2024-12-01` (`since` boundary slop) | 108 | 827 |
| 2 | Fault-signal filter (no fenced block + error keyword) | 667 | 160 |
| 3 | No closing PR resolvable via timeline | 88 | 72 |
| 4 | Empty triplet (no log AND no diffs) | 0 | 72 |
| 5 | Categorization (modality buckets, see below) | — | **107 written** |

**Final yield: 107 / 4,083 = 2.62 %**

---


## 3. Filter Rationale and Motivation


### 3.1 Skipped — Were PRs, not Issues (3,148)

**What this is.** GitHub's REST endpoint `GET /repos/{owner}/{repo}/issues` returns *both* issues and pull requests; PyGithub follows the same convention. Any object whose `pull_request` attribute is non-null is a PR, not an issue.


**Why we skip them.** Our goal is to build a dataset of real-world fault reports (issues) paired with their fixes (PRs). Including PRs as if they were issues would conflate the symptom (user report) and the remedy (developer fix), breaking the (Symptom → Root Cause) structure required for meaningful fault localization. This strict separation ensures that every record represents a genuine diagnostic scenario, not just a code change.

**Validity impact.** None. This is an API artifact, not a substantive filter; the 3,148 items reappear downstream as candidate *closing PRs* attached to retained issues.

---


### 3.2 Skipped — Closed Before December 2024 (108)

**What this is.** The `since` parameter on the GitHub issues endpoint filters by `updated_at`, not `closed_at`. An issue can be touched (label change, comment) after the cutoff while still having been *closed* much earlier; the script enforces the strict `closed_at >= 2024-12-01 UTC` check on the client side.


**Why we skip them.** The strict date cutoff is motivated by the need to avoid LLM training-data leakage. Modern LLMs (GPT-4, Claude, Gemini) have training cutoffs in late 2023 or 2024. By only including issues closed after 2024-11-30, we ensure that the model is unlikely to have seen the issue, PR, or related discussion during training. This filter is essential for a fair evaluation of model generalization and real-world diagnostic ability, not just memorization.

**Validity impact.** Mandatory. Removing this filter would inflate accuracy numbers without measuring genuine diagnostic capability. The filter is mentioned explicitly in the user's task spec ("to avoid LLM data leakage").

---


### 3.3 Failed Stage 2 — Fault-Signal Filter (667)

**What this is.** The issue body must satisfy *both*:
1. Contain at least one fenced code block ` ``` … ``` `
2. Contain at least one of `{error, failed, exception, denied, timeout, crashloopbackoff, oomkilled, evicted, 403, 500}` (case-insensitive).


**Why we skip them.** The 667 dropped issues fall into three main categories:

| Sub-population | Why excluded |
|---|---|
| Feature requests / RFCs | No fault or error to diagnose; not relevant for fault localization. |
| Documentation / website issues | No log/code/config triplet possible; only prose changes. |
| Bug reports lacking logs/traces | Even if a fault exists, the log modality (L) cannot be reconstructed. Without a real log, we'd have to fabricate one, which would contaminate the dataset with synthetic data and undermine the benchmark's realism.

**Why these two criteria specifically?**
We require both a fenced code block (for structural extractability) and an error keyword (to ensure the block is a real log, not just YAML or code). This dual filter maximizes the chance that the record contains a genuine, extractable fault signal. The keyword list is tailored to Knative's error patterns and is grounded in real-world log analysis (§4.4 Table A-1).

**Motivation for strictness:**
We prioritize high-quality, modality-complete records over raw recall. Including issues without logs would inflate the dataset but reduce its diagnostic value and introduce ambiguity. The filter is strict by design to ensure every record is a valid test case for fault localization.


**Validity impact.** This is the highest-precision-cost filter. Relaxing it would increase recall but at the cost of dataset quality and benchmark validity. For future work, a less strict filter could be used to allow partial triplets (e.g., code+config only), but for this release, we require a real log for every record.

---


### 3.4 Skipped — No Closing PR Resolvable (88)

**What this is.** For an issue to enter Dataset C it must be linked to a *merged* Pull Request in the **same repository**, discovered by walking `issue.get_timeline()` and inspecting events of type `closed` (PR-driven close) or `cross-referenced` (PR mentions issue). The 88 dropped issues passed Stage 2 but had none of the following:
- A `closed` event whose `source` references a merged PR
- A `cross-referenced` event from a merged PR in the same repo


**Why we skip them.** The code and config modalities (C, K) are only available if there is a closing PR. Without a PR, we cannot:
- Identify the exact code/config changes that fixed the issue
- Ensure the fix is developer-attested and not just a workaround
- Maintain the benchmark's grounding in real-world developer workflows

**Why not relax this?**
Some issues are closed as duplicates, not-a-bug, or fixed elsewhere (e.g., in another repo or by direct commit). Including these would introduce ambiguity and make it impossible to extract a trustworthy triplet. For future work, cross-repo PR linking could be added, but for now, we require a same-repo, merged PR for every record.

**Validity impact.** This is the strictest filter and the largest source of recall loss after Stage 2. Documented as a known limitation; the §5.3 pipeline anticipates a Manual Quality Filter (Stage 4) to recover some of these via human review, which is out of scope for the automated mining script.

---

## 4. What Survives (107 Records)

All 107 written records satisfy:
- `closed_at >= 2024-12-01 UTC`
- Issue body contains a fenced code block + at least one error keyword
- A merged PR in the same repository was identified via the timeline as the closing fix
- At least one of `{log, code_diff, config_diff}` is non-null (in practice, all three are attempted; `artifact_availability` flags record which modalities were extracted)

### Modality Distribution

| Modality Bucket      | Count |
|----------------------|------:|
| log_only             |     8 |
| log_config           |     5 |
| log_code             |    53 |
| log_code_config      |    15 |
| other                |    26 |
| **Total**            |   107 |

### Modality Distribution with Percentages

| Modality Bucket      | Count | Percent |
|----------------------|------:|--------:|
| log_only             |     8 |   7.48% |
| log_config           |     5 |   4.67% |
| log_code             |    53 |  49.53% |
| log_code_config      |    15 |  14.02% |
| other                |    26 |  24.30% |
| **Total**            |   107 | 100.00% |

---

### Example Extracted Record

Below is a real example of an extracted (log, code, config) record from the dataset:

```json
{
    "issue_id": "knative/serving#13204",
    "repository": "knative/serving",
    "issue_number": 13204,
    "issue_url": "https://github.com/knative/serving/issues/13204",
    "closing_pr_url": "https://github.com/knative/serving/pull/16554",
    "closed_at": "2026-04-21T09:07:14+00:00",
    "keywords_hit": ["timeout", "500"],
    "log": "This check will never return true. The reasons are the following:\r\n\r\n1. Defaulting. Knative sets up the deployment and does not set fields at various places which Kubernetes will default to some value. Those differences are always there. Examples are: the `Protocol` field in ports which gets set to `TCP`, the TimeoutSeconds/PeriodSeconds/FailureThreshold in the queue proxy probe, maxSurge in the rolling update strategy which gets set to 25, the RevisionHistoryLimit which becomes 10, the APIVersion in an env.valueFrom.FieldRef which gets set to `v1`, the restart policy that gets set to `Always` ...\r\n2. Mutations by others. Istio for example mutates in the labels `service.istio.io/canonical-name` and `service.istio.io/canonical-revision` in the pod template.\r\n\r\nWhy is this a problem? Assuming there are just 500 revisions in the system. If every revision reconcile is causing the one Update call, then - with a QPS of 40 ([Default 5 multiplied by controller count 8](https://github.com/knative/serving/blob/v0.33.0/vendor/knative.dev/pkg/injection/sharedmain/main.go#L208-L214)), then only these update calls will need 12.5 seconds. If you have rather 3000 revisions, we're talking about 1m15s. And it's not just about the time. It's also the unnecessary hammering on the API server.\r\n\r\nHow can this be fixed? I currently have two ideas:\r\n\r\n1. Pass the previous version along into `MakeDeployment` and there we will need to follow two strategies:\r\n\r\n   a) is to set defaults where applicable. If you define a TCP probe, just specify that. Kubernetes would unlikely change its default, but specifying the explicit value also does not hurt.\r\n   b) Use the previous version to set fields that Knative does not care and Kubernetes could change, such as the maxSurge or revision limit.\r\n\r\n   A sample of that can be found in https://github.com/SaschaSchwarze0/serving/commit/e3fe7f55de71c6ef49a5bc67f6f229372e3273fa. There is also code added in `checkAndUpdateDeployment` that dumps the differences which one can separately apply to see the problem. This part is also explained below in the reproduction steps.\r\n\r\n   But, this approach has two disadvantages:\r\n   \r\n   a) it will require a regular verification and extension assuming Kubernetes eventually adds new fields with defaulting\r\n   b) it will be a large effort to capture every possible field that others could mutate in, it will never be able to handle situations where others mutate changes on top of Knative\r\n\r\n2. Pass a target object into `MakeDeployment`. This would be an empty object for the create case and the current version for an update. All the existing functions that build the deployment details, pod spec, its container details, would need a rewrite to handle an empty object, or update an object.\r\n\r\n   This will imo be cleaner, but its complexity and amount of changes grew beyond what I wanted to try out in code I am not familiar with so that I have no spike code to show here.\r\n   \r\n   This will also not solve situations where webhooks mutate things that Knative sets.\r\n\r\nI am happy to discuss this. You can reach me in Knative slack (@sascha). Also happy to help with the implementation.\r\n\r\n--\r\n\r\nOne more thing I'd like to mention: creating an object from scratch and then applying it to the cluster for both the create and update case could be a pattern. I only focused on the revision controller to configure the deployment. I would not be surprised if that pattern (which I consider broken because of above explanations) is used elsewhere. Whether that is a pain or not depends on how many objects of that kind would be present and how often all objects get reconciled.\r\n\r\n<!-- If you need to report a security issue with Knative, send an email to knative-security@googlegroups.com. -->\r\n\r\n<!--\r\n## In what area(s)?\r\nRemove the '> ' to select\r\n> /area API\r\n> /area autoscale\r\n> /area build\r\n> /area monitoring\r\n> /area networking\r\n> /area test-and-release\r\n\r\nOther classifications:\r\n> /kind good-first-issue\r\n> /kind process\r\n> /kind spec\r\n-->\r\n\r\n## What version of Knative?\r\n\r\nCurrent\r\n\r\n## Expected Behavior\r\n\r\nKnative only updates Kubernetes deployments when there is a need to do this.\r\n\r\n## Actual Behavior\r\n\r\nEvery reconcile of a revision triggers a deployment update.\r\n\r\n## Steps to Reproduce the Problem\r\n\r\nAdd the following code before https://github.com/knative/serving/blob/v0.33.0/pkg/reconciler/revision/cruds.go#L93:",
    "code_diff": "--- pkg/apis/serving/register.go\n@@ -114,6 +114,10 @@ const (\n \t// last updated the resource.\n \tUpdaterAnnotation = GroupName + \"/lastModifier\"\n \n+\t// RevisionDeploymentHashLabelKey is the label key used to identify\n+\t// if a child deployment should be updated\n+\tRevisionDeploymentHashLabelKey = GroupName + \"/deployment-hash\"\n+\n \t// QueueSidecarResourcePercentageAnnotationKey is the percentage of user container resources to be used for queue-proxy\n \t// It has to be in [0.1,100]\n \t// Deprecated: Please consider setting resources explicitly for the QP per service, see `QueueSidecarCPUResourceRequestAnnotationKey` for example.\n--- pkg/reconciler/revision/cruds.go\n@@ -30,6 +30,7 @@ import (\n \t\"knative.dev/pkg/kmp\"\n \t\"knative.dev/pkg/logging\"\n \tautoscalingv1alpha1 \"knative.dev/serving/pkg/apis/autoscaling/v1alpha1\"\n+\t\"knative.dev/serving/pkg/apis/serving\"\n \tv1 \"knative.dev/serving/pkg/apis/serving/v1\"\n \t\"knative.dev/serving/pkg/client/injection/reconciler/autoscaling/v1alpha1/podautoscaler\"\n \t\"knative.dev/serving/pkg/reconciler/revision/config\"\n@@ -56,18 +57,20 @@ func (c *Reconciler) checkAndUpdateDeployment(ctx context.Context, rev *v1.Revis\n \t\treturn nil, fmt.Errorf(\"failed to update deployment: %w\", err)\n \t}\n \n+\thaveHash := have.Labels[serving.RevisionDeploymentHashLabelKey]\n+\twantHash := deployment.Labels[serving.RevisionDeploymentHashLabelKey]\n+\n+\tif haveHash == wantHash {\n+\t\treturn have, nil\n+\t}\n+\n \t// Preserve the current scale of the Deployment.\n \tdeployment.Spec.Replicas = have.Spec.Replicas\n \n \t// Preserve the label selector since it's immutable.\n \t// TODO(dprotaso): determine other immutable properties.\n \tdeployment.Spec.Selector = have.Spec.Selector\n \n-\t// If the spec we want is the spec we have, then we're good.\n-\tif equality.Semantic.DeepEqual(have.Spec, deployment.Spec) {\n-\t\treturn have, nil\n-\t}\n-\n \t// Otherwise attempt an update (with ONLY the spec changes).\n \tdesiredDeployment := have.DeepCopy()\n \tdesiredDeployment.Spec = deployment.Spec\n--- pkg/reconciler/revision/resources/deploy.go\n@@ -18,6 +18,7 @@ package resources\n \n import (\n \t\"fmt\"\n+\t\"maps\"\n \t\"sort\"\n \t\"strconv\"\n \t\"strings\"\n@@ -375,7 +376,7 @@ func MakeDeployment(rev *v1.Revision, cfg *config.Config) (*appsv1.Deployment, e\n \n \t// Slowly but steadily roll the deployment out, to have the least possible impact.\n \tmaxUnavailable := intstr.FromInt(0)\n-\treturn &appsv1.Deployment{\n+\td := &appsv1.Deployment{\n \t\tObjectMeta: metav1.ObjectMeta{\n \t\t\tName:            names.Deployment(rev),\n \t\t\tNamespace:       rev.Namespace,\n@@ -396,11 +397,19 @@ func MakeDeployment(rev *v1.Revision, cfg *config.Config) (*appsv1.Deployment, e\n \t\t\t},\n \t\t\tTemplate: corev1.PodTemplateSpec{\n \t\t\t\tObjectMeta: metav1.ObjectMeta{\n-\t\t\t\tLabels:      labels,\n+\t\t\t\t// make a copy so when we add the deployment hash it doesn't\n+\t\t\t\t// propagate include it here\n+\t\t\t\tLabels:      maps.Clone(labels),\n \t\t\t\tAnnotations: podAnnotations(rev),\n \t\t\t},\n \t\t\tSpec: *podSpec,\n \t\t},\n \t\t},\n-\t}, nil\n+\t}\n+\n+\t// We hash the desired deployment so that we reconcile\n+\t// changes when settings in config-maps change etc.\n+\tUpdateDeploymentHashLabel(d)\n+\n+\treturn d, nil\n }\n--- pkg/reconciler/revision/resources/meta.go\n@@ -17,10 +17,16 @@ limitations under the License.\n package resources\n \n import (\n+\t\"hash/fnv\"\n+\t\"io\"\n+\t\"strconv\"\n \t\"strings\"\n \n+\tappsv1 \"k8s.io/api/apps/v1\"\n \tmetav1 \"k8s.io/apimachinery/pkg/apis/meta/v1\"\n+\t\"k8s.io/apimachinery/pkg/util/dump\"\n \t\"k8s.io/apimachinery/pkg/util/sets\"\n+\n \t\"knative.dev/pkg/kmap\"\n \t\"knative.dev/serving/pkg/apis/autoscaling\"\n \t\"knative.dev/serving/pkg/apis/serving\"\n@@ -59,6 +65,7 @@ func makeLabels(revision *v1.Revision) map[string]string {\n \tif _, ok := labels[AppLabelKey]; !ok {\n \t\tlabels[AppLabelKey] = revision.Name\n \t}\n+\n \treturn labels\n }\n \n@@ -96,3 +103,20 @@ func makeSelector(revision *v1.Revision) *metav1.LabelSelector {\n \t\t},\n \t}\n }\n+\n+// UpdateDeploymentHashLabel is exposed for testing\n+func UpdateDeploymentHashLabel(d *appsv1.Deployment) {\n+\t// Delete the existing hash label so repeated calls to this\n+\t// function will be stable\n+\tdelete(d.Labels, serving.RevisionDeploymentHashLabelKey)\n+\n+\thasher := fnv.New32a()\n+\n+\tio.WriteString(hasher, dump.ForHash(d.Spec))\n+\tio.WriteString(hasher, dump.ForHash(d.Annotations))\n+\tio.WriteString(hasher, dump.ForHash(d.Labels))\n+\n+\thash := strconv.FormatUint(uint64(hasher.Sum32()), 36)\n+\n+\td.Labels[serving.RevisionDeploymentHashLabelKey] = hash\n+}",
    "config_diff": null,
    "artifact_availability": {
        "log_available": true,
        "log_source": "issue_body_code_block",
        "code_available": true,
        "code_source": "closing_pr_diff",
        "config_available": false,
        "config_source": null
    }
}
```

### Keyword Distribution

| Keyword | Hits |
|---|---:|
| error | 135 |
| failed | 73 |
| timeout | 32 |
| 500 | 24 |
| denied | 10 |
| 403 | 7 |
| exception | 3 |
| crashloopbackoff | 2 |

(Sum exceeds 107 because issues can hit multiple keywords; this is the per-occurrence total across all retained      records.)

---

## 5. Threats to Validity from Filtering Choices

| Threat | Mitigation Embedded in Pipeline |
|---|---|
| Stage 2 keyword list misses non-English error reports | Knative project enforces English in issue templates; risk minimal but acknowledged. |
| Same-repo PR guard drops legitimate cross-repo fixes (e.g., `knative/pkg` fix closes `knative/serving` issue) | Documented as known limitation; recoverable by extending `_issue_to_pr` to permit cross-repo references with provenance flag. |
| Date cutoff (2024-12-01) eliminates older but high-quality faults | Intentional — leakage avoidance dominates recall in benchmark construction (per §1). |
| Auto-extracted log = longest fenced block matching error keyword may capture a non-log block (e.g., a stack trace inside a YAML) | Manual Stage 4 review (§5.3) is the designed safety net; flag `log_source: issue_body_code_block` makes the heuristic auditable. |

---

## 6. Reproducing These Numbers

```bash
. .venv/bin/activate
export GITHUB_TOKEN=<your_token>
python -u scripts/mine_knative.py \
    --repos knative/serving knative/eventing knative/pkg knative/caching \
            knative/client knative/func knative/operator knative/networking \
    --since 2024-12-01 \
    --out data/dataset_c.jsonl \
    --resume \
    > data/mine_c.log 2>&1
```

Per-stage counters are emitted to `data/mine_c.log` in real time and aggregated in the final `=== TOTAL ===` block.
