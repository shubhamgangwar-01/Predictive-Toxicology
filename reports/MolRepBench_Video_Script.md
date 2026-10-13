# Video narration script — MolRepBench

**Shubham Gangwar · MS23BTECH11027** · Predictive Toxicology / AI in Drug Discovery

| Version | Words | Approx. runtime |
| --- | --- | --- |
| **Short cut** (skip every paragraph marked *Optional*) | 2,744 | **20–23 min** |
| **Full** (read everything) | 5,156 | **37–43 min** |

Record the short cut if your submission has a tight time limit; read everything for a full technical walkthrough. Numbers in the script come from the benchmark results.

**Delivery notes.** Acronyms are spelled out phonetically (R O C-A U C, S H A P) so they are read as letters, not words. Pause at each paragraph break. Slides 10 to 13 are reference tables — do not read them line by line.

---

## Slide 1 — Title

*~118 words · ~51 seconds*

Hello. I am Shubham Gangwar, roll number M S 23 B TECH 1 1 0 2 7, and this is my mini project: MolRepBench — Interpretable Toxicity and A D M E T Screening from Molecular Structure.

The project answers one practical question. To predict whether a molecule is toxic, should you describe that molecule using hand-designed chemical features, or let a neural network learn its own description from the molecular graph? The honest answer depends almost entirely on how much data you have — and in toxicology, you never have much.

I will show you the problem, the benchmark I built to answer that question, what it found, and the application I deployed on top of it.

---

## Slide 2 — The problem — why toxicity screening matters

*~125 words core (~54s) · ~183 words full (~78s)*

Let me start with why anyone should care.

Roughly nine out of ten drug candidates that enter clinical trials fail, and toxicity is a leading cause. The critical point is not that failures happen — it is when. A toxic liability found in a Phase Two trial has already consumed years of work and the entire preclinical budget. The same liability caught before the compound is synthesised costs essentially nothing.

> *Optional — cut for the short version.* Laboratory screening cannot close that gap. Tox21 tested about eight thousand compounds. The space of drug-like molecules is estimated at ten to the sixtieth. You cannot assay your way through that.

> *Optional — cut for the short version.* So the opportunity is clear: a model that reads a structure and flags risk in milliseconds lets a chemist triage thousands of candidates before committing to any.

But there is a catch, and the whole design of this project follows from it. A black-box risk score is not actionable. A chemist needs to know which part of the molecule is responsible, so they can redesign it — and needs to know when the model is guessing about chemistry it has never seen.

---

## Slide 3 — Problem statement and objectives

*~123 words · ~53 seconds*

That gives me my problem statement.

Build a system that predicts toxicity and A D M E T risk from a SMILES string. Establish, under an honest evaluation protocol, which molecular representation works better on small, imbalanced data. Explain every prediction at the level of chemical substructures. And deploy it as an application that flags compounds outside the model's domain of validity.

That gives four objectives. First, benchmark descriptor models against graph neural networks on four public datasets. Second, evaluate honestly — scaffold splits rather than random splits, and three metrics rather than one. Third, explain, using S H A P on fingerprints and attention from the graph networks. Fourth, deploy a Streamlit application returning a per-endpoint risk panel with a confidence flag.

---

## Slide 4 — Why this is harder than a normal ML benchmark

*~180 words · ~77 seconds*

Three things make this harder than a standard machine-learning benchmark.

The first is small data. After standardisation I have about 7,611 molecules for Tox21, 1,955 for the blood-brain barrier dataset, 1,380 for ClinTox, and 1,117 for solubility. Those are thousands, not millions. This is exactly the regime where data-hungry deep models historically struggle.

The second is severe class imbalance. In Tox21, the fraction of active — that is, toxic — compounds runs between about three and sixteen percent depending on the assay. A model that simply predicts "safe" for everything achieves ninety-five percent accuracy and is completely useless. So the choice of metric is not a detail here. It decides whether you can even see the problem.

The third is scaffold shift. In real drug discovery you are never asking about a molecule nearly identical to one you have already tested. You are asking about a new chemical series. A random train-test split quietly leaks close structural analogues into the test set and inflates every score you report. So I use scaffold splits, which I will explain in a moment.

---

## Slide 5 — Datasets

*~15 words core (~6s) · ~206 words full (~88s)*

Here are the four datasets, all from MoleculeNet, the standard benchmark suite in this field.

> *Optional — cut for the short version.* Tox21 is the big one: twelve separate toxicity assays covering nuclear receptor and stress response pathways. ClinTox contrasts F D A-approved drugs against compounds that failed clinical trials for toxicity. B B B P asks whether a compound crosses the blood-brain barrier. And E S O L is a regression task — predicting aqueous solubility. Together they cover toxicity and A D M E T, and both classification and regression.

> *Optional — cut for the short version.* One point I want to highlight, because it is the kind of thing that quietly ruins benchmarks. Raw MoleculeNet files contain salts, charged species, and the same compound written as several different SMILES strings. If you do not clean that up, the identical molecule can appear in both your training and your test set under two different strings, and your scores go up for no good reason.

> *Optional — cut for the short version.* So every molecule goes through a standardisation pipeline: parse it, keep only the largest organic fragment to strip salts and counter-ions, neutralise charges, canonicalise, and then merge duplicates — discarding any duplicates whose labels disagree. That removes between two and five percent of each file, and it removes a real source of inflated scores.

---

## Slide 6 — The three representations

*~107 words core (~46s) · ~254 words full (~109s)*

Now the central question of the project: how do you turn a molecule into numbers?

The first option is an E C F P 4 fingerprint. You walk outward from each atom, hash together the neighbourhood you find, and set a bit in a two-thousand-and-forty-eight-bit vector. Each bit therefore records the presence of a particular circular atom environment. Fingerprints need no training and are very fast — and crucially for this project, every bit maps back to a concrete, drawable substructure. That is what makes the explanations chemically meaningful later.

> *Optional — cut for the short version.* The second option is R D Kit descriptors: about two hundred and ten calculated properties — molecular weight, log P, polar surface area, ring counts. Decades of medicinal chemistry knowledge as explicit features. Fingerprints describe local substructure, descriptors describe the whole molecule, so the two are complementary and I benchmark both.

> *Optional — cut for the short version.* The third option is to use the molecular graph directly. Atoms become nodes with fifty-eight features each, bonds become edges with eleven. A graph neural network then passes messages between neighbouring atoms and learns its own representation, instead of being handed one.

So the question is simply: does the learned representation beat the hand-designed ones at this data scale?

> *Optional — cut for the short version.* I benchmark nine configurations. Random Forest, X G Boost and a neural network, each on both feature sets — that is six. Plus three graph networks: G C N, G A T, and Attentive F P, which was designed specifically for molecules and conveniently gives us attention weights we can read as an explanation.

---

## Slide 7 — Evaluation protocol

*~84 words core (~36s) · ~316 words full (~135s)*

This slide is the methodological heart of the project, so let me be precise.

Scaffold splitting works like this. You reduce every molecule to its Bemis-Murcko scaffold — its ring systems and the linkers joining them, with the side chains stripped off. You group molecules by that scaffold, and then you assign whole groups to a single subset. The result is that no scaffold appears in both training and test, so the test set consists entirely of chemical skeletons the model has never seen.

> *Optional — cut for the short version.* One finding here genuinely surprised me. Two scaffold splitting protocols are in common use: DeepChem's sorts scaffold groups largest first and is deterministic; Chemprop's shuffles them with a seed. I initially mixed the two across my seeds, and on B B B P they differed by about zero point two in R O C-A U C — a bigger gap than between any two models in the benchmark. The deterministic protocol pushes rare scaffolds into the test set and shifts the class balance substantially. So the protocol is now fixed across every run, and I report the two-protocol comparison separately. The lesson: in this field the split protocol can matter more than the model.

> *Optional — cut for the short version.* On metrics. I report three. R O C-A U C because it is the field standard — but it is insensitive to class imbalance by construction. P R-A U C, whose baseline equals the active rate, so it falls honestly when positives are rare. And M C C, which scores the actual yes-or-no decisions and is balanced across both classes. For M C C you need a threshold, and a default of zero point five is meaningless when three percent of compounds are active — so I tune the threshold on the validation split and then freeze it for test. Never on test.

> *Optional — cut for the short version.* Everything is run over three splits, and I report mean plus or minus standard deviation.

---

## Slide 8 — System architecture

*~0 words core (~0s) · ~144 words full (~62s)*

> *Optional — cut for the short version.* Briefly, the architecture.

> *Optional — cut for the short version.* A SMILES string comes in. It is standardised, then featurised three ways in parallel — fingerprints, descriptors, and a graph. Featurisation is cached, because computing descriptors over seven and a half thousand molecules is the slowest step in the pipeline.

> *Optional — cut for the short version.* Then the scaffold split, then all nine models are trained, then evaluation across the three metrics.

> *Optional — cut for the short version.* Two components sit alongside. The interpretability layer, which turns S H A P values and attention weights into per-atom maps. And the applicability domain, which decides whether a query compound is close enough to the training set for the prediction to mean anything.

> *Optional — cut for the short version.* Both feed the Streamlit app at the end. The codebase is organised as a package — data, features, models, metrics, pipeline, explain, applicability and predict — with separate scripts for each stage and unit tests covering standardisation, split disjointness, featurisation and metrics.

---

## Slide 9 — Results — the headline

*~159 words core (~68s) · ~371 words full (~159s)*

So, the headline result.

Descriptor models won on 4 of these 4 datasets, and graph networks on 0.

> *Optional — cut for the short version.* What you are seeing on this chart is the headline metric for every model on every dataset. Graph networks are in blue, descriptor models in grey. The error bars are the standard deviation across the three scaffold splits.

Two things to take from this. First, at this data scale, the hand-designed representations hold their own against learned graph representations. This is consistent with the broader literature — the advantage that learned representations enjoy in other fields comes from data volumes these benchmarks simply do not have.

Second, and I think more importantly — look at the size of those error bars. Three of those four margins sit inside one standard deviation of the split-to-split spread. On Tox21 the gap is eleven thousandths against a spread of thirty-one; on B B B P it is four thousandths against forty. Only ClinTox separates the two families cleanly. So the claim I am willing to defend is not that descriptors are better — it is that graph networks do not beat them at this data scale, while costing considerably more to train.

> *Optional — cut for the short version.* There is one more thing in this chart that a family-versus-family framing hides, and it is arguably the most useful finding for anyone building a pipeline. The two descriptor representations do not behave alike. On E S O L, the solubility task, R D Kit descriptors and the graph networks together take the top six places, with R M S E between zero point eight and zero point nine nine. All three E C F P models sit far behind at one point four eight to one point six five. That is a gap several times larger than anything separating the leaders. The reason is chemically obvious once you see it: solubility is governed by bulk physical properties — logP, polar surface area, molecular weight — which R D Kit computes directly, and which a binary substructure fingerprint simply cannot encode. The graph networks learn most of it from structure. E C F P cannot represent it at all. The lesson is that you match the representation to the endpoint, not to the pipeline.

---

## Slide 10-13 — Per-dataset results tables

*~35 words core (~15s) · ~185 words full (~79s)*

These next slides give the full numbers for each dataset, so they are there for reference rather than for reading aloud line by line.

A few things worth pointing out as we go through them.

> *Optional — cut for the short version.* On Tox21, the largest and most imbalanced dataset, notice how much lower P R-A U C is than R O C-A U C for every model. That gap is the class imbalance showing through, and it is the honest picture.

> *Optional — cut for the short version.* On ClinTox, several models score poorly in M C C terms despite respectable R O C-A U C. ClinTox is small, and its first task is about ninety-four percent positive, so there is very little signal for a model to learn. I am reporting that rather than hiding it, because a benchmark that only reports its successes is not a benchmark.

> *Optional — cut for the short version.* On B B B P, this is the dataset where the split protocol mattered most, as I described earlier.

> *Optional — cut for the short version.* And on E S O L, the regression task, performance is measured in R M S E and R squared instead — lower R M S E is better here.

---

## Slide 14 — The split protocol moved scores more than the model

*~167 words core (~72s) · ~390 words full (~167s)*

This slide shows the single largest effect I measured in the whole project — and it is not a model choice.

I ran identical models under both scaffold splitting protocols, changing nothing else. The blue bars are the Chemprop balanced protocol; the grey bars are the DeepChem deterministic one. The mean difference is zero point one two three in R O C-A U C, and on B B B P it reaches zero point one six eight. Hold that against the architecture result on the same dataset: the best descriptor model beats the best graph network there by four thousandths. The evaluation protocol is worth roughly forty times the architecture.

> *Optional — cut for the short version.* The effect is also strongly dataset-dependent, and the pattern tells you something. B B B P, at just under two thousand molecules, moves by zero point one six eight. ClinTox, at fourteen hundred, moves by zero point one four seven. But Tox21, at seventy-six hundred molecules, moves by only zero point zero four two. With more scaffolds available, which particular ones land in the test set matters proportionally less. Protocol sensitivity is a small-data pathology — which is exactly the regime nearly every toxicity benchmark lives in.

> *Optional — cut for the short version.* Importantly, this is not a bug in either protocol. I audited both for leakage by measuring, for every test molecule, its maximum fingerprint similarity to the training set. Near-duplicate rates stay between zero and four percent in every split, so neither protocol is leaking. What differs is distance. The deterministic protocol pushes the rare, structurally unusual tail of the dataset into the test set — its test molecules sit further from training, and on B B B P the class balance shifts from eighty-two percent positive in training to fifty-four percent in test. Under the balanced protocol those same two numbers are seventy-six and seventy-eight percent — within two points of each other. It asks a harder question. Reassuringly, its numbers reproduce the figures commonly published for these datasets, which tells me the implementation is right.

The implication is uncomfortable. If a researcher changed protocol and attributed the resulting jump to their architecture, they would reach a confident and completely wrong conclusion. So in this project the protocol is fixed across every run, recorded in every result row, and there is a unit test that fails if the two are ever mixed again.

---

## Slide 15 — Why metric choice changes the conclusion

*~170 words core (~73s) · ~228 words full (~98s)*

This slide, for me, is the most important one in the project.

On the left, each point is one of the twelve Tox21 assays, plotted against how rare its active compounds are. The grey points are R O C-A U C and the blue points are P R-A U C. Notice what happens as you move left, towards the rarer assays. R O C-A U C stays in a fairly respectable band. P R-A U C falls off a cliff.

> *Optional — cut for the short version.* That is not a defect in the metric — it is the metric doing its job. P R-A U C's baseline is the active rate itself, so when only four percent of compounds are active, a P R-A U C of zero point two means that most of the compounds your model ranks highly are not actually active.

So you can have a model reported at R O C-A U C zero point seven five, which sounds perfectly usable, and which in practice would send a chemist chasing mostly false positives. And M C C completes the picture: several configurations here achieve decent ranking while their actual yes-or-no calls sit close to chance.

The practical conclusion is this. R O C-A U C alone — which is still the most commonly quoted number in this literature — is not sufficient evidence that a toxicity screening model is usable.

---

## Slide 16-18 — Interpretability — SHAP evidence

*~181 words core (~78s) · ~439 words full (~188s)*

Now, explanations — and this is where the project produced both its most satisfying result and its most interesting complication.

For the fingerprint models I use Tree S H A P, which gives an exact contribution for each of the two thousand fingerprint bits. Because each bit corresponds to a circular atom environment, I spread that bit's value across the atoms of that environment, giving a score for every atom. Red pushes predicted risk up, blue pushes it down.

> *Optional — cut for the short version.* The test of whether this works is whether it independently recovers chemistry a medicinal chemist already knows. On the active compounds, it does. For bisphenol A, a well-documented endocrine disruptor scored at zero point eight three, every single atom carries positive attribution and the red concentrates on both phenol rings — exactly the groups that mimic estradiol's hydroxyls and drive receptor binding. For estradiol itself, scored at zero point eight nine, attribution concentrates on the phenolic A-ring, the textbook estrogen receptor pharmacophore. Its two largest bits, at plus zero point seven two and plus zero point six four, both describe that aromatic hydroxyl environment.

Aspirin is the more interesting case, and I am showing it as it came out rather than as the clean negative control I picked it to be. Aspirin has no nuclear receptor liability, and qualitatively the map behaves: the ester and carboxyl regions render blue, and the single largest-magnitude bit is negative. But the atom weights sum to only plus zero point one six, and the probability comes out at zero point five two. That is not a clearance. The benzene ring aspirin shares with genuine estrogen receptor ligands contributes positive evidence that the ester and acid groups do not fully cancel.

> *Optional — cut for the short version.* I want to be clear that I regard this as a useful result rather than an embarrassing one. On an endpoint whose pharmacophore is an aromatic ring carrying a hydroxyl, a fingerprint model reading substructures in isolation will find part of that motif in a molecule that does not bind. The attribution does not hide that — it makes the reason for the borderline score legible, instead of leaving you with an unexplained zero point five two.

> *Optional — cut for the short version.* The next slide shows why this matters for a screening tool specifically. Because every fingerprint bit maps back to a drawable substructure, I can show the individual environments carrying the attribution. The dominant one for bisphenol A carries a S H A P value of plus zero point eight eight and is the hydroxyl-bearing aromatic environment. That is not a saliency map a chemist has to interpret — it is a specific motif they can design away.

---

## Slide 19 — Interpretability — attention, and what did not generalise

*~274 words core (~117s) · ~409 words full (~175s)*

Now the complication, and this is the part of the project where I had to throw away my own explanation.

I expected Attentive F P's readout attention to give a second, independent account of each prediction. What I found is that it does — on one of the four fitted models, and not on the other three.

This slide shows the same molecule, estradiol, under all four Attentive F P models. Same architecture, same extraction code, same molecule, four different training sets. The ClinTox model is sharply selective: maximum-to-minimum attention ratios between five and thirty-two across the example compounds, with normalised entropies around zero point eight to zero point nine six, where one point zero is perfectly uniform. The Tox21, B B B P and E S O L models are not: ratios between one point zero three and one point one zero, and entropies of zero point nine nine nine nine or above. Those three readouts attend to every atom in the molecule almost exactly equally. They carry no information at all.

> *Optional — cut for the short version.* I checked whether this was my bug. The coefficients come from the final readout timestep and sum to one across each molecule's atoms, exactly as the softmax requires. So it is a property of the fitted models, not an extraction error.

What I want to be honest about is that this inverts the explanation I gave earlier in the project. An earlier partial run saw flat attention on ClinTox, and I attributed it to ClinTox being the smallest dataset in the benchmark — eleven hundred training molecules, not enough pressure for the readout to become selective. I predicted that a model trained on Tox21, with roughly six times the data, would become sharp. The completed run on the G P U inverted it exactly. ClinTox is now the sharp one, and Tox21 is flat. Training set size does not explain the pattern.

> *Optional — cut for the short version.* So the conclusion I can actually defend is weaker than my hypothesis, but more useful. Attention concentration is a property of one particular fitted model, reached through one particular optimisation run. It is not a property of the Attentive F P architecture that you can count on in advance. And to make that point sharper — the ClinTox model, the one with the beautifully concentrated attention, is also the weakest classifier in that comparison, at R O C-A U C zero point seven zero. Sharp attention is not even evidence of a good model.

---

## Slide 20 — Interpretability — the hazard, and what the two methods can claim

*~268 words core (~115s) · ~406 words full (~174s)*

That leads to the thing I think is genuinely worth the audience's attention, which is how this failure hides itself.

Look at the colour scale on the previous slide. The conventional way to draw an attention map normalises each molecule against its own minimum and maximum — it stretches whatever spread exists to fill the full colour range. Under that convention, all four of those panels render as vivid, structured, completely convincing pictures. Including the three that encode nothing, because min-max normalisation faithfully amplifies variation in the fourth decimal place. I drew them against an absolute reference instead, where full colour means three times uniform attention, and three of the four panels go blank.

That is the hazard in one sentence: an uninformative attention map does not look uninformative.

So the application measures before it shows. Every attention map is scored for its max-min ratio and its normalised entropy before being rendered, and a map that fails the test is displayed with an explicit warning pointing the user to the S H A P attribution, rather than being presented as an explanation. The figures in my report are generated the same way, which is why three of those four panels are deliberately blank instead of quietly left out.

One last caveat, which this project happens to have direct evidence for. These two methods answer different questions. S H A P is counterfactual — how would the prediction change if this substructure were absent. Attention is internal to the forward pass — where the model looked, which is related to, but not the same as, what drove the output.

> *Optional — cut for the short version.* And on the one model where I can read both, they disagree. For bisphenol A, S H A P picks out the two phenol rings, which is the correct pharmacophore. The ClinTox attention instead concentrates on the central quaternary carbon joining them — structurally central, causally irrelevant. Both maps are sharp, both are internally consistent, and only one of them agrees with the chemistry. Where the two do converge, as they do on estradiol's aromatic A-ring, I trust the explanation more.

> *Optional — cut for the short version.* But neither is a causal claim about biology. Both explain the model. Neither explains the toxicity. An attribution is enough to guide a redesign, and enough to notice when a model is keying on something chemically absurd. It is not evidence of a mechanism, and it should not be handed to a regulator as though it were.

---

## Slide 21 — Applicability domain — knowing when you do not know

*~140 words core (~60s) · ~171 words full (~73s)*

A Q S A R model is only valid inside the chemical space it was trained on. Most deployed models ignore this completely and will return a confident-looking number for literally any input.

So every prediction here carries a domain badge, computed from two independent checks.

The first is structural. I compute the mean Tanimoto similarity between the query compound and its five nearest training molecules. Rather than picking an arbitrary cut-off, I compute that same statistic leave-one-out across the whole training set, and use the fifth and twenty-fifth percentiles of that distribution as my thresholds. That way the criterion adapts automatically to how structurally diverse each dataset actually is.

> *Optional — cut for the short version.* The second check is physicochemical: what fraction of the R D Kit descriptors fall outside the range seen in training. That catches compounds that look structurally familiar but are physically extreme.

Together they produce three levels. In domain — predictions carry normal weight. Borderline — interpret with care. And out of domain — treat every number on the screen as unreliable.

---

## Slide 22 — The deployed application

*~90 words core (~39s) · ~157 words full (~67s)*

This is the application. You give it a SMILES string, and it returns a risk panel covering sixteen endpoints: the twelve Tox21 assays, the two ClinTox tasks, blood-brain barrier penetration, and predicted solubility. Each endpoint shows its probability, its tuned cut-off, and the resulting call, sorted so the highest risks appear first.

> *Optional — cut for the short version.* Above that sits the confidence badge, along with the three most similar training compounds, shown explicitly so the user can judge the comparison for themselves rather than trusting my similarity number.

> *Optional — cut for the short version.* The second tab answers "why". It shows the S H A P atom heat map for whichever endpoint you select, draws the most influential fingerprint environments as individual substructures, and shows the attention map beside them.

And the third tab ships the entire benchmark inside the application. That was a deliberate decision. A screening tool that shows predictions without showing their measured reliability is inviting exactly the overconfidence this project set out to address.

---

## Slide 23 — What the benchmark showed

*~308 words core (~132s) · ~451 words full (~193s)*

So, the conclusions.

Descriptor models won on 4 of these 4 datasets, and graph networks on 0. At this data scale, learned graph representations do not displace well-established fingerprints and descriptors — though three of those four margins sit inside one standard deviation, so what I am really claiming is parity at considerably lower training cost, not superiority.

But the most consequential finding was methodological. The evaluation protocol moved scores by zero point one two three R O C-A U C on average, and zero point one six eight on B B B P — a dataset where the two model families differ by four thousandths. That is a factor of about forty. Combined with split-to-split variance that often exceeds the gap between model families, it suggests a substantial share of reported progress in this literature may come from evaluation choices rather than from modelling advances.

> *Optional — cut for the short version.* B B B P makes that concrete in a way I did not expect. The nine model configurations span a range of zero point zero three two R O C-A U C, from zero point nine one one to zero point nine four three. The average split-to-split standard deviation within any single configuration is zero point zero four five — larger than the entire spread between the best and the worst model. On that dataset my benchmark does not rank the architectures at all. It establishes that they are indistinguishable. Had I run a single split and reported a winner, I would have produced a confident ordering with nothing behind it.

> *Optional — cut for the short version.* Scaffold splits also cost every model real performance compared to the random-split numbers usually quoted. That is not a flaw — it is the price of asking the question that actually matters.

Metric choice changes the conclusion. On the rarest assays, R O C-A U C suggests a usable model while P R-A U C and M C C show the decisions are near chance.

And the variance across splits is large — often larger than the gap between model families. That has a direct implication for how this literature should be read: single-split comparisons on datasets of this size cannot reliably distinguish between architectures, and a reported improvement of a point or two may be indistinguishable from split noise.

And finally, on interpretability: S H A P delivered and attention did not, at least not reliably. S H A P recovered the estrogen receptor pharmacophore on every example I tried. Attention was informative on one of the four fitted models and flat on the other three, with no way to predict in advance which. That is why the deployed application measures every attention map's concentration before it will show it as an explanation.

---

## Slide 24 — Limitations and future work

*~80 words core (~34s) · ~305 words full (~131s)*

I want to be straightforward about the limitations.

> *Optional — cut for the short version.* The benchmark datasets carry their own biases — ClinTox, for instance, labels regulatory approval status, which is not the same thing as toxicity, since drugs fail for efficacy and commercial reasons too.

> *Optional — cut for the short version.* Hyperparameters are fixed per model family rather than tuned per dataset. That is what keeps the comparison fair, but tuning would probably help the graph networks more than the tree ensembles, since tree ensembles are famously insensitive to it. So some part of the gap I measured may not survive careful tuning.

> *Optional — cut for the short version.* The split-protocol experiment covered the six descriptor configurations on all four datasets and G C N on three of them, rather than the full nine-model grid — the effect was consistent in sign and size across every family I measured, but G A T and Attentive F P were only run under the balanced protocol.

> *Optional — cut for the short version.* Three seeds bound the variance only loosely. Attention is not causal explanation, and as I showed, it was informative on only one of four fitted models. And the applicability domain is a similarity heuristic, not a calibrated uncertainty estimate — it tells you a compound is unusual, not the probability that this particular prediction is wrong.

For future work, the most promising direction is self-supervised pretraining. The core limitation facing the graph networks here is label scarcity, and pretraining on large unlabelled molecular corpora — the approach taken by ChemBERTa, GROVER and Mol C L R — is the established route around it. That would test directly whether my conclusion is a statement about architectures, or merely about data volume. I suspect it is mostly about data volume.

> *Optional — cut for the short version.* Beyond that: calibrated uncertainty through ensembles or conformal prediction, multi-task transfer from Tox21's twelve assays into the smaller endpoints, and ultimately prospective validation on compounds with newly measured experimental results.

---

## Slide 25 — Closing

*~120 words · ~51 seconds*

To summarise.

I built MolRepBench: a benchmark of nine models across three molecular representations and four datasets, evaluated under a protocol designed to be honest rather than flattering. It found that hand-designed representations remain competitive at this data scale, that the evaluation protocol can matter more than the model, and that reporting a single metric would have led to the wrong conclusion.

On top of that benchmark sits a working application where every prediction comes with the substructure that drove it, and an explicit statement of whether the compound is one the model is entitled to have an opinion about.

Thank you for watching. I am Shubham Gangwar, roll number M S 23 B TECH 1 1 0 2 7.

---
