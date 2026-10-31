# M3: ASCEND

**Adaptive Self-supervised Continual Expert Network for Domain-Incremental Wi-Fi Sensing**

- **Course:** ECE 310 Wireless Communication, Monsoon Semester 2026
- **Category:** Research Project
- **Team:** <group_name>
- **Members:** Dhvanit Mahesuriya (AU2440039), <other members and IDs>
- **Submission date:** <date>


## Project in brief

Wi-Fi sensing uses Channel State Information (CSI) to recognise human activities without cameras. A model trained in one room often fails in another, and training it on a new room makes it forget the old ones. ASCEND learns rooms one after another. It uses a frozen self-supervised backbone, one adapter and one specialist per room, and a router that picks the right room at test time.

## Main results (single run)

| Dataset | ASCEND | SAMoE-C (closest method) |
|---|---|---|
| MM-Fi (average final accuracy) | 90.30% | 85.04% |
| MM-Fi (average forgetting) | 1.19% | 7.27% |
| Widar3.0 (average final accuracy) | 66.75% | 61.92% |

The smallest variant, ASCEND-ULite, uses 35.4 MMACs, about 82% less than SAMoE-C, and gives up 4.90 points of accuracy on MM-Fi.

## Limitations

- Results come from single runs, so there are no error bars.
- The ablation was run on Widar3.0 only.
- Label-efficiency tests and multi-seed runs are not done yet.
