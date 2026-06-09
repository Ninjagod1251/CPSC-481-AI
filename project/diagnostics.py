from probability4e import BayesNet, enumeration_ask

T, F = True, False


class Diagnostics:
    """Diagnose between TB, Lung cancer, and Bronchitis using the 'Asia'
    Bayesian network and exact inference (enumeration)."""

    def __init__(self):
        # Nodes are listed parents-before-children (topological order),
        # which BayesNet requires. Each CPT entry is P(node = True | parents).
        self.net = BayesNet([
            ('Asia', '', 0.01),
            ('Smoking', '', 0.5),
            ('TB', 'Asia', {T: 0.05, F: 0.01}),
            ('Cancer', 'Smoking', {T: 0.1, F: 0.01}),
            ('Bronchitis', 'Smoking', {T: 0.6, F: 0.3}),
            # Deterministic OR: TBorC is True iff TB or Cancer is True.
            ('TBorC', 'TB Cancer',
             {(T, T): 1.0, (T, F): 1.0, (F, T): 1.0, (F, F): 0.0}),
            ('Xray', 'TBorC', {T: 0.99, F: 0.05}),
            ('Dyspnea', 'TBorC Bronchitis',
             {(T, T): 0.9, (T, F): 0.7, (F, T): 0.8, (F, F): 0.1}),
        ])

    def diagnose(self, asia, smoking, xray, dyspnea):
        # Map GUI strings -> Boolean evidence. "NA" => omit the variable so
        # enumeration treats it as hidden/unobserved (NOT observed-False).
        evidence = {}
        if asia != 'NA':
            evidence['Asia'] = (asia == 'Yes')
        if smoking != 'NA':
            evidence['Smoking'] = (smoking == 'Yes')
        if xray != 'NA':
            evidence['Xray'] = (xray == 'Abnormal')
        if dyspnea != 'NA':
            evidence['Dyspnea'] = (dyspnea == 'Present')

        # Posterior P(disease = True | evidence) for each disease.
        diseases = ['TB', 'Cancer', 'Bronchitis']
        probs = {d: enumeration_ask(d, evidence, self.net)[True] for d in diseases}

        best = max(probs, key=probs.get)
        return [best, probs[best]]
