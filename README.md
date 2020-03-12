# CS269 NLP - FinalProject

K-Word Question Attack on Reading Comprehension Systems

Slide: https://docs.google.com/presentation/d/1el33w2EV1fSomhh3IK3iXCko3pvYSAEgoZ9GA13Aftk/edit?usp=sharing

# Contents
1. *generate_question.py*: The script to generate k-word questions.
2. *dev-v1.1.json*: The origianl SQuAD v1.1 dataset from https://rajpurkar.github.io/SQuAD-explorer/.
3. *sample-correct-incorrect.py*: The modified version of the SQuAD v1.1 evaluation script with the ability to sample `new_incorrect` and `both_correct` results.
    * `new_incorrect`: The `(Context, OriginalQuestion, NewQuestion, OriginalAnswer, NewAnswer)` tuple where `OriginalAnswer` is correct and `NewAnswer` is **incorrect**.
    * `both_correct`: The `(Context, OriginalQuestion, NewQuestion, OriginalAnswer, NewAnswer)` tuple where **both** `OriginalAnswer` and `NewAnswer` are **correct**.
4. *albert-results/*: Contains the output of *generate_question.py*, the [ALBERT](https://github.com/kamalkraj/ALBERT-TF2.0) base model predictions, and the output of *sample-correct-incorrect.py*.
5. *Distillbert results/*: Contains the predictions summary of [DistillBERT](https://github.com/oliverproud/DistilBERT-SQuAD).

# Reference
See the reference page in the slide.
