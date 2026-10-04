# 🔐 Exploration of Quantum Money

**Max Landon, Karandeep Shoker, Nick Huynh**
University of British Columbia — November 2025

## Abstract

Quantum money aims to provide an unforgeable, physically unclonable form of currency which could revolutionize cryptography and digital transactions. In this paper, we provide background on the foundational ideas that make quantum banknotes theoretically viable, including a discussion of Wiesner's pioneering work on quantum money and the no-cloning theorem, which underpins its security.

We also examine the ongoing open problem of designing publicly verifiable quantum money, a system that would allow anyone to authenticate a quantum banknote without access to secret keys, thereby enabling practical, cash-like quantum transactions. In particular, we investigate proposed constructions based on hidden subspaces, while also briefly touching on the now-disproven lattice-based approach.

## 1 Background: Quantum Information and Unforgeability

### 1.1 From Classical to Quantum Information

In classical computation, information is encoded in bits that take definite values $0$ or $1$. A bit-string can be copied arbitrarily many times without loss of information. In quantum computation, however, a qubit exists in a superposition

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1, \quad \alpha, \beta \in \mathbb{C} \tag{1}
$$

Measuring this qubit in a basis collapses its state, irreversibly destroying the superposition. Therefore, observation inevitably changes the system. Any attempt to "copy" or "read" an unknown quantum state without full knowledge of its basis inevitably introduces detectable errors.

### 1.2 The No-Cloning Theorem

The impossibility of perfect copying was formalized by Wootters and Zurek in 1982. A simple modern proof follows [[Mer07]](#ref-mer07):

Suppose there existed a unitary operator $U$ that cloned arbitrary states:

$$
U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle. \tag{2}
$$

It follows from linearity that:

$$
U(a|\psi\rangle + b|\phi\rangle)|0\rangle = aU|\psi\rangle|0\rangle + bU|\phi\rangle|0\rangle = a|\psi\rangle|\psi\rangle + b|\phi\rangle|\phi\rangle \tag{3}
$$

But also:

$$
U(a|\psi\rangle + b|\phi\rangle)|0\rangle = \big(a|\psi\rangle + b|\phi\rangle\big)\big(a|\psi\rangle + b|\phi\rangle\big) = a^2|\psi\rangle|\psi\rangle + b^2|\phi\rangle|\phi\rangle + ab|\psi\rangle|\phi\rangle + ab|\phi\rangle|\psi\rangle \tag{4}
$$

These two statements are contradictory unless $a$ or $b$ is $0$. This proves that there cannot exist such a $U$ that can clone arbitrary states.

This fundamental property — the absence of a universal quantum copier — forms the security foundation of all quantum money protocols.

Physically, the theorem follows from the linearity of quantum evolution (unitarity in Hilbert space): copying one basis vector correctly would force inconsistencies for superpositions of basis states.

Note that it **is** possible to determine / clone a quantum state provided you know which orthogonal basis it belongs to.

For example, the states $|0\rangle$ and $|1\rangle$ are orthogonal to each other and form an orthogonal basis. So, if we were given some state $|\psi\rangle$ and we knew $|\psi\rangle \in \lbrace |0\rangle, |1\rangle\rbrace $, then we **could** clone $|\psi\rangle$:

In this case, we would simply use $U = CNOT\_{1,2}$

Since:

$$
CNOT\_{1,2} \cdot |0\rangle|0\rangle = |0\rangle|0\rangle \qquad \text{and} \qquad CNOT\_{1,2} \cdot |1\rangle|0\rangle = |1\rangle|1\rangle \tag{5}
$$

### 1.3 Conjugate Bases and Measurement Disturbance

Quantum mechanics allows different choices of measurement bases that are **mutually unbiased**, meaning that if a state is prepared in one basis, measurements in the other yield completely random outcomes. For qubits, two canonical examples are

$$
Z\text{-basis}: \lbrace |0\rangle, |1\rangle\rbrace , \qquad X\text{-basis}: \lbrace |+\rangle, |-\rangle\rbrace , \qquad |\pm\rangle = \frac{1}{\sqrt{2}}\big(|0\rangle \pm |1\rangle\big).
$$

If a qubit is prepared in $|+\rangle$ but measured in the $Z$-basis, the result is uniformly random where each outcome occurs with probability $1/2$. This measurement also collapses the state to $|0\rangle$ or $|1\rangle$, destroying the original superposition. Measuring in the wrong basis therefore yields no useful information and irreversibly disturbs the state, a core feature underlying the security of quantum money schemes.

## 2 Wiesner's Private-Key Quantum Money

### 2.1 Original Conjugate-Coding Scheme

Stephen Wiesner [[Wie83]](#ref-wie83) proposed the first unforgeable money system based on quantum physics. Each banknote consists of a classical serial number $s$ and an $n$-qubit quantum register:

$$
|\psi\_s\rangle = \bigotimes\_{i=1}^{n} |b\_i\rangle\_{\theta\_i} \tag{6}
$$

where each bit $b\_i \in \lbrace 0, 1\rbrace $ and each basis choice $\theta\_i \in \lbrace Z, X\rbrace $ is chosen uniformly at random by the issuing bank.

- If $\theta\_i = Z$:
  - If $b\_i = 0$: $|b\_i\rangle\_{\theta\_i} = |0\rangle$
  - If $b\_i = 1$: $|b\_i\rangle\_{\theta\_i} = |1\rangle$
- If $\theta\_i = X$:
  - If $b\_i = 0$: $|b\_i\rangle\_{\theta\_i} = |+\rangle$
  - If $b\_i = 1$: $|b\_i\rangle\_{\theta\_i} = |-\rangle$

The bank maintains a secret database mapping each serial number to its corresponding list of bases and bit values $(\theta\_i, b\_i)$.

The actual "banknote" would consist of the serial number and the quantum $n$-qubit register:

$$
(s, |\psi\_s\rangle) \tag{7}
$$

Using a quantum banknote would involve physically moving the quantum state $|\psi\_s\rangle$ to the recipient as well as telling them the serial number. The recipient could periodically go to the bank to verify if their quantum notes are valid.

### 2.2 Verification Process

To authenticate a note, the bank measures each qubit in its original basis $\theta\_i$ and checks that all measurement outcomes match the stored $b\_i$. If every qubit yields the correct result, the note is valid, otherwise it is rejected.

Mathematically, verification is a projective measurement using projectors

$$
P\_s = |\psi\_s\rangle\langle\psi\_s|, \qquad I - P\_s,
$$

where $P\_s$ projects onto the correct subspace. Any counterfeit state $|\phi\rangle$ not equal to $|\psi\_s\rangle$ will pass with probability

$$
P\_{\text{forge}} = |\langle \psi\_s | \phi \rangle|^2 < 1.
$$

## 3 Security Analysis of Wiesner's Scheme

A counterfeiter lacks the secret basis choices $\theta\_i$. To attempt cloning, they must choose a measurement basis for each qubit:

- Measuring in the correct basis succeeds with probability $1$.
- Measuring in the wrong basis yields a successful outcome with probability $50\\%$ and disturbs the state. This disruption will cause any subsequent measurements to also succeed with probability $50\\%$.

### 3.1 Pure Cloning Strategies

The simplest strategy is for the attacker to just guess each of the $n$ qubit states. Since there are $4$ possible states for each qubit, and the bank will falsely accept $2$ of those states $50\\%$ of the time, this strategy will succeed with probability:

$$
\left( \frac{1}{4} + \frac{1}{2} \cdot \frac{1}{2} \right)^n = \left( \frac{1}{2} \right)^n \tag{8}
$$

Another (only slightly) more sophisticated attack will start with some (valid) $n$-qubit quantum state $|\psi\_s\rangle$ which the attacker will measure and try to create a copy of. This will leave the attacker with $2$ banknotes, both of which they hope will be verified successfully by the bank.

For each qubit $|b\_i\rangle\_{\theta\_i}$ in this state, the attacker will simply measure the qubit in one of the $2$ bases $\lbrace Z, X\rbrace $, chosen at random. They will prepare a new qubit in the state measured and use that qubit in their cloned banknote. We naturally have $2$ cases:

- ($\tfrac{1}{2}$ probability) **The attacker measures in the correct basis.** If the attacker guesses the basis correctly, they will reproduce the state of $|b\_i\rangle\_{\theta\_i}$. When the bank verifies the $2$ banknotes for this qubit, it will succeed with probability $1$.
- ($\tfrac{1}{2}$ probability) **The attacker measures in the incorrect basis.** If the attacker measures in the wrong basis, they will incorrectly reproduce the state of $|b\_i\rangle\_{\theta\_i}$. But this doesn't necessarily mean that the attacker will be caught. When the bank is verifying the attacker's note, it will measure in the correct basis, which will succeed with probability $\tfrac{1}{2}$. Since the attacker wants both notes to be valid (otherwise they will not have profited), the attacker will succeed for this qubit with probability $\tfrac{1}{2} \cdot \tfrac{1}{2} = \tfrac{1}{4}$.

Overall the probability of the bank successfully verifying the attacker's cloned banknotes is:

$$
\left( \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot \frac{1}{4} \right)^n = \left( \frac{5}{8} \right)^n \tag{9}
$$

For a reasonably large value of $n$, this is still an astronomically small chance. Therefore, this attack is not effective.

In fact, Molina et al. [[MVW12]](#ref-mvw12) showed that the optimal attack of this nature (where the attacker is only given access to a valid banknote), would only succeed with probability $(3/4)^n$.

As this probability still decays exponentially with $n$, we can consider this and therefore any pure cloning strategy ineffective.

### 3.2 Effective Cloning Strategy [[BNSU14]](#ref-bnsu14)

As we've shown, it is very hard for an attacker to re-create a quantum state that will pass verification from the bank. However, in the previously mentioned strategies, the attacker does not use the bank's verification procedure to help perform the cloning.

In 2014, Nagaj et al. devised a simple strategy that uses the bank verification procedure of a valid note to perfectly learn its quantum state (therefore being able to clone it easily) with arbitrarily high chance of success.

We begin with a known valid quantum banknote with serial number $s$ and quantum state $|\psi\_s\rangle$. The attack is executed one qubit at a time: given $|b\_i\rangle\_{\theta\_i}$, we try to learn $b\_i$ and $\theta\_i$.

#### 3.2.1 Checking for $|+\rangle$

The first part of this attack determines whether our unknown qubit $|b\_i\rangle\_{\theta\_i} = |+\rangle$. We start with a control qubit $|0\rangle$. Then:

1. We rotate the control qubit by some small angle $\epsilon$.
2. We apply a CNOT gate with the control qubit on our unknown qubit.
3. We give the unknown qubit to the bank to verify.
4. We repeat steps 1–3 $N = \dfrac{\pi}{2\epsilon}$ times.
5. We measure the value of our control qubit. If it outputs $1$, we know that $|b\_i\rangle\_{\theta\_i} = |+\rangle$. Otherwise, it does not.

To show that this works, we will analyze each possible state of $|b\_i\rangle\_{\theta\_i}$.

**Case 1: $|b\_i\rangle\_{\theta\_i} = |+\rangle$.** State starts out as $|0\rangle|+\rangle$.

Once we rotate our control qubit we have:

$$
(\cos\epsilon\\,|0\rangle + \sin\epsilon\\,|1\rangle)\\,|+\rangle
$$

Applying $CNOT\_{1,2}$ will not change the state since $|+\rangle$ is unaffected by the $NOT$ operation.

When we give our unknown qubit to the bank to verify, it will always approve it, because it has not changed from $|+\rangle$.

After $N$ iterations of the algorithm, the control qubit will have rotated by $\tfrac{\pi}{2}$ and will therefore be $= |1\rangle$. Measuring this will output $1$ with probability $100\\%$.

**Case 2: $|b\_i\rangle\_{\theta\_i} = |-\rangle$.** State starts out as $|0\rangle|-\rangle$.

Once we rotate our control qubit we have:

$$
(\cos\epsilon\\,|0\rangle + \sin\epsilon\\,|1\rangle)\\,|-\rangle \xrightarrow{CNOT\_{1,2}} (\cos\epsilon\\,|0\rangle - \sin\epsilon\\,|1\rangle)\\,|-\rangle
$$

When we give our unknown qubit to the bank to verify, it will always approve it, because it has not changed from $|-\rangle$.

In this case, since the control qubit is now $|0\rangle$ rotated by $-\epsilon$ radians, the next iteration will return the qubit back toward $|0\rangle$. Then, as long as we chose $N$ to be an even number, the control qubit will be $|0\rangle$ after $N$ iterations. Measuring this will output $0$ with probability $100\\%$.

**Case 3: $|b\_i\rangle\_{\theta\_i} = |0\rangle$.** State starts out as $|0\rangle|0\rangle$.

Once we rotate our control qubit we have:

$$
(\cos\epsilon\\,|0\rangle + \sin\epsilon\\,|1\rangle)\\,|0\rangle \xrightarrow{CNOT\_{1,2}} \cos\epsilon\\,|00\rangle + \sin\epsilon\\,|11\rangle
$$

In this and the next case, when we give the bank our unknown qubit, it will be $|1\rangle$ (incorrect) with a small probability $\sin^2\epsilon$. With very high probability, though, $|0\rangle$ will be measured which will collapse the state of our control qubit back to $|0\rangle$.

After $N$ iterations, assuming the bank always measures $|0\rangle$, our control qubit will be $|0\rangle$ as well. Measuring this will output $0$ with probability $100\\%$.

**Case 4: $|b\_i\rangle\_{\theta\_i} = |1\rangle$.** State starts out as $|0\rangle|1\rangle$.

Once we rotate our control qubit we have:

$$
(\cos\epsilon\\,|0\rangle + \sin\epsilon\\,|1\rangle)\\,|1\rangle \xrightarrow{CNOT\_{1,2}} \cos\epsilon\\,|01\rangle + \sin\epsilon\\,|10\rangle
$$

When we give the bank our unknown qubit, it will be $|0\rangle$ (incorrect) with a small probability $\sin^2\epsilon$. With very high probability, though, $|1\rangle$ will be measured which will collapse the state of our control qubit back to $|0\rangle$.

After $N$ iterations, assuming the bank always measures $|1\rangle$, our control qubit will be $|0\rangle$. Measuring this will output $0$ with probability $100\\%$.

Overall, we can see that in case 1, we measure a $1$ at the end of the procedure, and in the other $3$ cases, we measure a $0$. This means we have a reliable way to determine if the unknown qubit is $|+\rangle$ or not. If we measure $1$, we can stop now, satisfied that we know the value of $|b\_i\rangle\_{\theta\_i}$. Otherwise, we continue.

#### 3.2.2 Checking for $|-\rangle$

Now that we know that our unknown qubit $|b\_i\rangle\_{\theta\_i}$ is not $|+\rangle$, we will apply a similar process to determine if $|b\_i\rangle\_{\theta\_i} = |-\rangle$.

In fact, the only difference is that instead of applying $CNOT\_{1,2}$, we apply a "controlled-$NOT$." Which will NOT and negate the affected qubit if the control qubit is $|1\rangle$, and do nothing otherwise.

The effect this will have is making the case with $|-\rangle$ behave like the $|+\rangle$ case in the previous step. The $|0\rangle$ and $|1\rangle$ cases will behave like they did in the previous steps. Overall, we see that with $|-\rangle$, we output $1$, and with $|0\rangle$ or $|1\rangle$, we output $0$. At this point, if we measure $1$, we can be confident that $|b\_i\rangle\_{\theta\_i} = |-\rangle$. Otherwise, we do the final check.

#### 3.2.3 Checking for $|0\rangle$ or $|1\rangle$

Now that we know our unknown qubit $|b\_i\rangle\_{\theta\_i} \notin \lbrace |+\rangle, |-\rangle\rbrace $, we know $\theta\_i = Z$ and we can simply measure in that basis to determine the value of $|b\_i\rangle\_{\theta\_i}$.

#### 3.2.4 Probability of Success [[BNSU14, p.6]](#ref-bnsu14)

The worst case for the attacker is the case where all the qubits in $|\psi\rangle$ are $\in \lbrace |0\rangle, |1\rangle\rbrace $. In this case, for each qubit, the chance of the bank successfully verifying the qubit for all $N$ trials is:

$$
Pr(\text{Success}\_i) = (1 - \sin^2\epsilon)^N \\;\ge\\; \left(1 - \frac{\pi^2}{4N^2}\right)^N \\;\ge\\; 1 - N \cdot \frac{\pi^2}{4N^2} = 1 - \frac{\pi^2}{4N} \tag{10}
$$

For each qubit, we might have to perform $2$ such checks (one for $|+\rangle$ and one for $|-\rangle$). So overall:

$$
Pr(\text{Success}) \\;\ge\\; \left(1 - \frac{\pi^2}{4N}\right)^{2n} \\;\ge\\; 1 - \frac{\pi^2 n}{2N} \tag{11}
$$

So we can set $N$ to be as large as we want to avoid chances of detection.

#### 3.2.5 What This Attack Means for Quantum Money

By performing frequent measurements, the bank's verification projects the money state back to its basis states, effectively "freezing" it while our control qubit accumulates information about the relative phase.

To successfully execute this attack, the attacker would need to have the bank verify their note many times. The bank might be suspicious if the same person is verifying their note over and over again. There are ways to reduce the suspicion of the attacker though, like passing the note to multiple people, for example, making this attack theoretically viable.

The existence of this attack strategy demonstrates the need for any quantum money issuing authority (bank) to re-issue a new banknote instead of giving back a banknote whenever it is verifying one.

## 4 Publicly Verifiable Quantum Money

The practical constraint of needing the bank to re-issue a banknote every time someone wants to verify it severely limits the appeal of quantum money, as the advantages over classic schemes seem to disappear.

This dilemma motivated the open problem of **publicly verifiable quantum money**, quantum banknotes which can be verified by anyone without needing access to secret information. The work was pioneered by Aaronson and Christiano [[AC12]](#ref-ac12) and developed further by Zhandry [[Zha25]](#ref-zha25).

Various candidate constructions for publicly verifiable schemes have been proposed and subsequently broken. For example, a recent lattice-based approach attempted to encode banknotes using Gaussian-weighted superpositions over integer lattices. However, this scheme was effectively disproven by Liu, Montgomery, and Zhandry [[LMZ23]](#ref-lmz23), who demonstrated that the verification mechanism was insufficient; an adversary could exploit linear combinations of short lattice vectors to forge valid banknotes, violating the unforgeability requirement.

In the rest of this paper, we will discuss hidden subspaces, a promising scheme for publicly verifiable quantum money.

### 4.1 Hidden Subspaces

Aaronson and Christiano [[AC12]](#ref-ac12) introduced the **hidden-subspace** framework for quantum money, where each banknote is represented by a quantum state defined by a random linear subspace. The mint privately chooses a subspace

$$
S \subseteq \mathbb{F}\_2^n
$$

of dimension $n/2$ and prepares the uniform superposition

$$
|S\rangle = \frac{1}{\sqrt{|S|}} \sum\_{x \in S} |x\rangle.
$$

The quantum banknote $|S\rangle$ is a uniform superposition over all bitstrings in the subspace $S$, so its information is carried by the coherence across every vector in that subspace. Any measurement that attempts to extract a specific basis string disrupts this coherence and destroys the banknote.

To allow public verification while keeping $S$ hidden, the mint publishes two efficiently computable membership oracles:

$$
f\_S(x) = \begin{cases} 1, & \text{if } x \in S, \\\\ 0, & \text{otherwise,} \end{cases} \qquad\qquad f\_{S^\perp}(x) = \begin{cases} 1, & \text{if } x \in S^\perp, \\\\ 0, & \text{otherwise.} \end{cases}
$$

These black-box functions reveal only whether a given basis vector lies in $S$ or its orthogonal complement $S^\perp$, without exposing any basis information. Verification proceeds by checking that a candidate state lies in $S$ using $f\_S$, then applying a Hadamard transform and checking that the transformed state lies in $S^\perp$ using $f\_{S^\perp}$. Only the uniform superposition $|S\rangle$ passes both tests, ensuring public verifiability and security against forgery.

### 4.2 Mathematical Verification and Uniqueness Proof

Let $|\psi\rangle$ be a candidate banknote state. Following the "Hidden Subspace Mini-Scheme" defined in Section 5 of [[AC12]](#ref-ac12), the security of the protocol relies on the fact that a valid banknote must satisfy two rigid constraints regarding its support in the primal (standard) and dual (Hadamard) bases.

**Constraint 1: Primal Support (Standard Basis).** The state must have zero probability amplitude outside the subspace $S$.

$$
\text{Supp}(|\psi\rangle) \subseteq S \iff \text{proj}\_S(|\psi\rangle) = |\psi\rangle \tag{12}
$$

where $\text{proj}\_S = \sum\_{x \in S} |x\rangle\langle x|$ is the projection onto the hidden subspace.

**Constraint 2: Dual Support (Hadamard Basis).** The Hadamard transform of the state must have zero probability amplitude outside the dual subspace $S^\perp$.

$$
\text{Supp}(H^{\otimes n}|\psi\rangle) \subseteq S^\perp \iff \text{proj}\_{S^\perp}(H^{\otimes n}|\psi\rangle) = H^{\otimes n}|\psi\rangle \tag{13}
$$

where $\text{proj}\_{S^\perp} = \sum\_{y \in S^\perp} |y\rangle\langle y|$.

#### 4.2.1 Proof of Completeness

We verify that the uniform superposition $|\psi\rangle = |S\rangle = \dfrac{1}{\sqrt{|S|}}\sum\_{x \in S}|x\rangle$ strictly satisfies both constraints.

**Verifying Constraint 1:** We apply the projection operator $\text{proj}\_S$ directly to the candidate state. Utilizing the linearity of the projection and the orthonormality of the computational basis ($\langle k|x\rangle = \delta\_{kx}$):

$$
\text{proj}\_S|\psi\rangle = \left( \sum\_{k \in S} |k\rangle\langle k| \right) \left( \frac{1}{\sqrt{|S|}} \sum\_{x \in S} |x\rangle \right) \tag{14}
$$

$$
= \frac{1}{\sqrt{|S|}} \sum\_{x \in S} \sum\_{k \in S} |k\rangle \\, \delta\_{kx} \tag{15}
$$

$$
= \frac{1}{\sqrt{|S|}} \sum\_{x \in S} |x\rangle = |\psi\rangle \tag{16}
$$

Thus, Constraint 1 is strictly satisfied by construction.

**Verifying Constraint 2:** We compute the Hadamard transform of the candidate state to determine its support in the dual basis. By substituting the definition $H^{\otimes n}|x\rangle = \dfrac{1}{\sqrt{2^n}} \sum\_{y \in \lbrace 0,1\rbrace ^n} (-1)^{x \cdot y}|y\rangle$ and swapping the order of summation, we isolate the interference term for each basis vector $|y\rangle$:

$$
H^{\otimes n}|\psi\rangle = \frac{1}{\sqrt{|S|}} \sum\_{x \in S} H^{\otimes n}|x\rangle = \frac{1}{\sqrt{|S|\\,2^n}} \sum\_{y \in \lbrace 0,1\rbrace ^n} \underbrace{\left( \sum\_{x \in S} (-1)^{x \cdot y} \right)}\_{C\_y} |y\rangle \tag{17}
$$

We analyze the inner interference term, $C\_y = \sum\_{x \in S} (-1)^{x \cdot y}$, in two rigorous cases:

- **Case $y \in S^\perp$:** By definition, $x \cdot y = 0$ for all $x \in S$. The phase is $(-1)^0 = 1$, so the sum becomes $C\_y = \sum\_{x \in S} 1 = |S|$.

- **Case $y \notin S^\perp$:** Since $y$ is not orthogonal to $S$, there exists at least one $x^* \in S$ such that $x^* \cdot y = 1$. Then define a new function $f: x \mapsto x + x^*\ \forall x \in S$. The image of this function will remain in $S$ because $x, x^* \in S$. Also, since addition is its own inverse in $\mathbb{F}\_2^n$, $f(f(x)) = x$. Since $f$ is its own inverse, it is a bijection, which means it perfectly maps every element in $S$ to another element in $S$, each element appearing in the image exactly once.

  So we can rewrite $C\_y$ as the following without loss of generality:

  $$
  C\_y = \sum\_{x \in S} (-1)^{(x + x^*) \cdot y} = \sum\_{x \in S} (-1)^{x \cdot y + x^* \cdot y} = \sum\_{x \in S} (-1)^{x \cdot y} (-1)^1 = -\sum\_{x \in S} (-1)^{x \cdot y} = -C\_y
  $$

  This implies $2C\_y = 0$, and thus $C\_y = 0$.

Substituting these results back into Eq. (17), all terms where $y \notin S^\perp$ are eliminated. To simplify the normalization constant, we utilize the fundamental dimension property of subspaces in $\mathbb{F}\_2^n$ (implied by the definitions in Section 2 of [[AC12]](#ref-ac12) and the Rank–Nullity Theorem): since $\dim(S) + \dim(S^\perp) = n$, it follows that $|S| \cdot |S^\perp| = 2^n$.

$$
H^{\otimes n}|\psi\rangle = \frac{|S|}{\sqrt{|S|\\,2^n}} \sum\_{y \in S^\perp} |y\rangle = \sqrt{\frac{|S|^2}{|S|\\,2^n}} \sum\_{y \in S^\perp} |y\rangle = \sqrt{\frac{|S|}{2^n}} \sum\_{y \in S^\perp} |y\rangle = \frac{1}{\sqrt{|S^\perp|}} \sum\_{y \in S^\perp} |y\rangle = |S^\perp\rangle \tag{18}
$$

The transformed state is exactly the uniform superposition over $S^\perp$. By definition, this state lies entirely within the support of $\text{proj}\_{S^\perp}$, so Constraint 2 is strictly satisfied.

#### 4.2.2 Proof of Uniqueness

We prove that the uniform superposition is the **unique** state satisfying both constraints.

Any state satisfying Constraint 1 has support strictly inside $S$, so $|\psi\rangle = \sum\_{x \in S} \alpha\_x |x\rangle$. Let its Hadamard transform be $H^{\otimes n}|\psi\rangle = \sum\_{y \in \lbrace 0,1\rbrace ^n} \beta\_y |y\rangle$.

To find $\alpha\_x$, we use the property $H^2 = I$. Applying $H^{\otimes n}$ to the transformed state must recover $|\psi\rangle$:

$$
|\psi\rangle = H^{\otimes n}\left( \sum\_{y \in \lbrace 0,1\rbrace ^n} \beta\_y |y\rangle \right) = \sum\_{y \in \lbrace 0,1\rbrace ^n} \beta\_y \left( \frac{1}{\sqrt{2^n}} \sum\_{x \in \lbrace 0,1\rbrace ^n} (-1)^{x \cdot y} |x\rangle \right) \tag{19}
$$

We rearrange the summation to group terms by the basis vector $|x\rangle$:

$$
|\psi\rangle = \sum\_{x \in \lbrace 0,1\rbrace ^n} |x\rangle \underbrace{\left( \frac{1}{\sqrt{2^n}} \sum\_{y \in \lbrace 0,1\rbrace ^n} \beta\_y (-1)^{x \cdot y} \right)}\_{\alpha\_x} \tag{20}
$$

Applying Constraint 1, will limit the summation to $x \in S$:

$$
|\psi\rangle = \sum\_{x \in S} |x\rangle \underbrace{\left( \frac{1}{\sqrt{2^n}} \sum\_{y \in \lbrace 0,1\rbrace ^n} \beta\_y (-1)^{x \cdot y} \right)}\_{\alpha\_x} \tag{21}
$$

By matching coefficients and enforcing $\beta\_y = 0$ outside $S^\perp$ to satisfy Constraint 2, we obtain:

$$
\alpha\_x = \frac{1}{\sqrt{2^n}} \sum\_{y \in S^\perp} \beta\_y (-1)^{x \cdot y} \tag{22}
$$

Since $x \in S$ and $y \in S^\perp$, orthogonality implies $x \cdot y = 0$ and $(-1)^{x \cdot y} = 1$. The expression simplifies to:

$$
\alpha\_x = \frac{1}{\sqrt{2^n}} \sum\_{y \in S^\perp} \beta\_y \tag{23}
$$

The right-hand side is independent of $x$, implying all amplitudes $\alpha\_x$ are identical. The only normalized state with identical amplitudes over $S$ is the uniform superposition $|S\rangle$. Thus, the valid money state must be unique.

### 4.3 Implementing the Membership Oracle with Polynomials

To move from a theoretical framework to a practical currency, we must replace the abstract "black-box" oracles with an explicit cryptographic construction. The verifier needs a classical description of the membership function $f\_S$ that allows efficient verification but prevents the recovery of the hidden subspace $S$.

Aaronson and Christiano propose instantiating these oracles using systems of multivariate polynomials.

#### 4.3.1 Hiding the Subspace

The mint hides the secret subspace $S$ by generating a system of $m$ multivariate polynomials $P = \lbrace p\_1, \ldots, p\_m\rbrace $ over the field $\mathbb{F}\_2$. These polynomials are constructed to satisfy the **vanishing condition**:

$$
p\_i(x) = 0 \qquad \forall x \in S,\ \forall i \in \lbrace 1, \ldots, m\rbrace  \tag{24}
$$

The mint publishes $P$ as the "public key" for the primal subspace. Similarly, it generates a second system $Q = \lbrace q\_1, \ldots, q\_m\rbrace $ that vanishes on the dual subspace $S^\perp$ to serve as the public key for the dual filter.

#### 4.3.2 Verification and Hardness

The verification procedure replaces the oracle query with polynomial evaluation. To verify Constraint 1, the user computes $p\_i(x)$ for all $i$. The state can be verified and accepted as belonging to $S$ only if it is a common zero of the system $P$, or $p\_i(x) = 0$ for all $i$.

The security of this construction relies on the computational hardness of solving random systems of multivariate equations. While evaluating the polynomials takes polynomial time $O(m \cdot n^d)$, recovering the underlying subspace $S$ from the coefficients is conjectured to require exponential time. Specifically, the scheme assumes that no polynomial-time quantum algorithm can find a generating set for $S$ given $P$, provided the degree of the polynomials is $d \ge 4$ [[AC12, p. 38]](#ref-ac12).

#### 4.3.3 Introducing Noise

To further secure the scheme against algebraic attacks (such as linearization), the construction can be augmented with noise. The mint includes a small fraction of "decoy" polynomials in $P$ that do **not** vanish on $S$. The verification condition is relaxed to require that $x$ satisfies a large majority (e.g., $(1-\epsilon)m$) of the equations. This forces a counterfeiter to solve the **Learning Parity with Noise** problem, which is widely considered hard even for quantum adversaries [[AC12, p. 37]](#ref-ac12).

### 4.4 Feasibility

A counterfeiter receives one valid note $|S\rangle$ and public access to the oracles $f\_S$ and $f\_{S^\perp}$, but no information about a basis for $S$ or $S^\perp$. To forge a new note, the adversary must either clone $|S\rangle$, which is prohibited by the no-cloning theorem, or prepare a fresh copy without knowing the hidden subspace.

From the mint's perspective, the scheme remains practical:

- Random subspaces and the corresponding states $|S\rangle$ can be generated efficiently.
- The oracles $f\_S$ and $f\_{S^\perp}$ can be implemented using pseudorandom classical hash functions such as multivariate polynomials.
- Verification requires only two oracle checks and a Hadamard transform, making it efficient and scalable.

Together, these properties make the hidden-subspace framework one of the first plausible and efficient realizations of **public-key quantum money**.

## 5 Limitations

While publicly verifiable quantum money solves the online attack problem, it faces significant implementation barriers:

- **Cryptographic Instability of Obfuscation:** The security of the hidden subspace scheme relies on securely hiding the subspace description using multivariate polynomials. This rests on unproven cryptographic assumptions; specifically, low-degree polynomial obfuscations have already been broken by classical algebraic attacks, and no proven secure obfuscation method currently exists [[AC12, p. 37]](#ref-ac12).
- **The Decoherence Barrier:** Unlike quantum communication, quantum money requires long-term storage. Current quantum states decohere on the scale of microseconds, whereas a practical currency requires coherence for months or years, a technology that remains theoretically challenging.
- **Verification Rigidity:** The hidden subspace protocol requires strict rank-1 projective measurements to check if a state is **exactly** in subspace $S$. In a noisy physical world, this lack of tolerance would likely cause valid banknotes to be rejected due to minor errors, while relaxing the tolerance could introduce loopholes for counterfeiters [[AC12, p. 5,6]](#ref-ac12).

## 6 Conclusion

In this work, we explored the theoretical foundations and current state of quantum money. Starting with Wiesner's original conjugate coding scheme, we demonstrated how the No-Cloning Theorem provides security against counterfeiting, albeit with the practical limitation of requiring centralized verification. We analyzed the vulnerability of private-key schemes to adaptive attacks, highlighting the necessity for a shift toward publicly verifiable quantum money.

Our investigation into public-key schemes focused on the Hidden Subspace framework by Aaronson and Christiano. By leveraging the duality between a subspace and its orthogonal complement under the Hadamard transform, this approach theoretically enables secure, offline verification without secret keys. However, we found that the implementation of such schemes remains an open challenge.

Ultimately, while quantum money promises a physically unforgeable currency with decentralized transactions, significant barriers remain. Future research must address the instability of obfuscation methods and the physical challenge of long-term quantum memory before quantum banknotes can become a practical reality.

## References

1. <a id="ref-ac12"></a>**[AC12]** Scott Aaronson and Paul Christiano. *Quantum money from hidden subspaces.* In Proceedings of the forty-fourth annual ACM symposium on Theory of computing, pages 41–60, 2012. [pp. 7, 8, 10, 11, 12]
   [https://arxiv.org/abs/1203.4740](https://arxiv.org/abs/1203.4740)

2. <a id="ref-bnsu14"></a>**[BNSU14]** Aharon Brodutch, Daniel Nagaj, Or Sattath, and Dominique Unruh. *An adaptive attack on Wiesner's quantum money.* arXiv preprint arXiv:1404.1507, 2014. [pp. 4, 7]
   [https://arxiv.org/abs/1404.1507](https://arxiv.org/abs/1404.1507)

3. <a id="ref-lmz23"></a>**[LMZ23]** Jiahui Liu, Hart Montgomery, and Mark Zhandry. *Another round of breaking and making quantum money: How to not build it from lattices, and more.* In Annual International Conference on the Theory and Applications of Cryptographic Techniques, pages 611–638. Springer, 2023. [p. 7]
   [https://arxiv.org/abs/2211.11994](https://arxiv.org/abs/2211.11994)

4. <a id="ref-mer07"></a>**[Mer07]** N. David Mermin. *Quantum computer science: an introduction.* Cambridge University Press, 2007. [p. 1]
   [https://www.cambridge.org/9780521876582](https://www.cambridge.org/9780521876582)

5. <a id="ref-mvw12"></a>**[MVW12]** Abel Molina, Thomas Vidick, and John Watrous. *Optimal counterfeiting attacks and generalizations for Wiesner's quantum money.* In Conference on Quantum Computation, Communication, and Cryptography, pages 45–64. Springer, 2012. [p. 4]
   [https://arxiv.org/abs/1202.4010](https://arxiv.org/abs/1202.4010)

6. <a id="ref-wie83"></a>**[Wie83]** Stephen Wiesner. *Conjugate coding.* ACM Sigact News, 15(1):78–88, 1983. [p. 2]
   [https://doi.org/10.1145/1008908.1008920](https://doi.org/10.1145/1008908.1008920)

7. <a id="ref-zha25"></a>**[Zha25]** Mark Zhandry. *Quantum money from abelian group actions.* TheoretiCS, 4, 2025. [p. 7]
   [https://arxiv.org/abs/2307.12120](https://arxiv.org/abs/2307.12120)
