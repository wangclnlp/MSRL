import re
from typing import List, Optional

from swift.rewards import ORM, orms


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''


def _normalize_label(text: str) -> str:
    text = str(text).strip()
    answer = _extract_tag(text, 'answer')
    if answer:
        text = answer
    return text.strip().upper()


def _normalize_task(text: str) -> str:
    text = str(text).strip().lower()
    task = _extract_tag(text, 'type')
    if task:
        text = task
    text = text.replace('_', ' ').replace('-', ' ')
    return ' '.join(text.split())


class MSRLFormatReward(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL | re.IGNORECASE))
            has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL | re.IGNORECASE))
            rewards.append(1.0 if has_think and has_answer else 0.0)
        return rewards


class MSRLAccuracyReward(ORM):

    def __call__(self,
                 completions,
                 solution: Optional[List[str]] = None,
                 chosen: Optional[List[str]] = None,
                 label: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        targets = solution or chosen or label
        if targets is None:
            return [0.0] * len(completions)

        rewards = []
        for completion, target in zip(completions, targets):
            pred = _normalize_label(completion)
            gold = _normalize_label(target)
            rewards.append(1.0 if pred == gold and pred != '' else 0.0)
        return rewards


class MSRLTaskReward(ORM):

    def __call__(self, completions, task_type: Optional[List[str]] = None, **kwargs) -> List[float]:
        if task_type is None:
            return [0.0] * len(completions)

        rewards = []
        for completion, gold_task in zip(completions, task_type):
            pred_task = _normalize_task(completion)
            gold_task = _normalize_task(gold_task)
            rewards.append(0.2 if pred_task == gold_task and pred_task != '' else 0.0)
        return rewards


orms['msrl_format_reward'] = MSRLFormatReward
orms['msrl_accuracy_reward'] = MSRLAccuracyReward
orms['msrl_task_reward'] = MSRLTaskReward
