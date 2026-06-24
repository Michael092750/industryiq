<!-- image -->

## NIST Trustworthy and Responsible AI NIST AI 600-1

## Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile

This publication is available free of charge from: https://doi.org/10.6028/NIST.AI.600-1

<!-- image -->

## NIST Trustworthy and Responsible AI NIST AI 600-1

## Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile

This publication is available free of charge from: https://doi.org/10.6028/NIST.AI.600-1

July 2024

<!-- image -->

U.S. Department of Commerce Gina M. Raimondo, Secretary

National Institute of Standards and Technology

Laurie E. Locascio, NIST Director and Under Secretary of Commerce for Standards and Technology About AI at NIST : The National Institute of Standards and Technology (NIST) develops measurements, technology, tools, and standards to advance reliable, safe, transparent, explainable, privacy -enhanced, and fair artificial intelligence (AI) so that its full commercial and societal benefits can be realized without harm to people or the planet. NIST, which has conducted both fundamental and applied work on AI for more than a decade, is also helping to fulfill the 2023 Executive Order on Safe, Secure, and Trustworthy AI. NIST established the U.S. AI Safety Institute and the companion AI Safety Institute Consortium to continue the efforts set in motion by the E.O. to build the science necessary for safe, secure, and trustworthy development and use of AI.

Acknowledgments: This report was accomplished with the many helpful comments and contributions from the community, including the NIST Generative AI Public Working Group, and NIST staff and guest researchers: Chloe Autio, Jesse Dunietz, Patrick Hall, Shomik Jain, Kamie Roberts, Reva Schwartz, Martin Stanley, and Elham Tabassi.

## NIST Technical Series Policies

[Copyright, Use, and Licensing Statements NIST Technical Series Publication Identifier Syntax](https://www.nist.gov/nist-research-library/nist-technical-series-publications-author-instructions#pubid)

## Publication History

Approved by the NIST Editorial Review Board on 07 -25-2024

## Contact Information

[ai-inquiries@nist.gov](mailto:ai-inquiries@nist.gov)

National Institute of Standards and Technology Attn: NIST AI Innovation Lab, Information Technology Laboratory 100 Bureau Drive (Mail Stop 8900) Gaithersburg, MD 20899 -8900

## Additional Information

Additional information about this publication and other NIST AI publications are available at https://airc.nist.gov/Home.

Disclaimer : Certain commercial entities, equipment, or materials may be identified in this document in order to adequately describe an experimental procedure or concept. Such identification is not intended to imply recommendation or endorsement by the National Instit ute of Standards and Technology, nor is it intended to imply that the entities, materials, or equipment are necessarily the best available for the purpose. Any mention of commercial, non -profit, academic partners, or their products, or reference s is for information only; it is not intended to imply endorsement or recommendation by any U.S. Government agency.

## Table of Contents

| 1.                                                                                                                                       | Introduction ..............................................................................................................................................1   | Introduction ..............................................................................................................................................1   |
|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.                                                                                                                                       | Overview of Risks Unique to or Exacerbated by GAI.....................................................................2                                        | Overview of Risks Unique to or Exacerbated by GAI.....................................................................2                                        |
| 3.                                                                                                                                       | Suggested Actions to Manage GAI Risks.........................................................................................12                               | Suggested Actions to Manage GAI Risks.........................................................................................12                               |
| Appendix A. Primary GAI Considerations ...............................................................................................47 | Appendix A. Primary GAI Considerations ...............................................................................................47                       | Appendix A. Primary GAI Considerations ...............................................................................................47                       |

## 1. Introduction

This document is a cross-sectoral profile of and companion resource for the AI Risk Management Framework (AI RMF 1.0 ) for Generative AI, 1 pursuant to President Biden's Executive Order (EO) 14110 on Safe, Secure, and Trustworthy Artificial Intelligence. 2 The AI RMF was released in January 2023, and is intended for voluntary use and to improve the ability of organizations to incorporate trustworthiness considerations into the design, development, use, and evaluation of AI products, services, and systems.

A profile is an implementation of the AI RMF functions, categories, and subcategories for a specific setting, application, or technology -in this case, Generative AI (GAI) -based on the requirements, risk tolerance, and resources of the Framework user. AI RMF profile s assist organizations in deciding how to best manage AI risks in a manner that is well -aligned with their goals, considers legal/regulatory requirements and best practices, and reflects risk management priorities. Consistent with other AI RMF p rofiles, this profile offers insights into how risk can be managed across various stages of the AI lifecycle and for GAI as a technology.

As GAI covers risks of models or applications that can be used across use cases or sectors, this document is an AI RMF cross -sectoral profi le. Crosssectoral profiles can be used to govern, map, measure, and manage risks associated with activities or business processes common across sectors, such as the use of large language models (LLMs), cloud -based services, or acquisition.

This document defines risks that are novel to or exacerbated by the use of GAI. After introducing and describing these risks, the document provides a set of suggested actions to help organizations govern, map, measure, and manage these risks.

1 EO 14110 defines Generative AI as 'the class of AI models that emulate the structure and characteristics of input data in order to generate derived synthetic content. This can include images, videos, audio, text, and other digital content.' While not all GAI is derived from foundation models, for purposes of this document, GAI generally refers to generative foundation models . The foundation model subcategory of 'dual -use foundation models' is defined by EO 14110 as 'an AI model that is trained on broad data; generally uses self-supervision; contains at least tens of billions of parameters; is applicable across a wide range of contexts.'

2 This profile was developed per Section 4.1(a)(i)(A) of EO 14110, which directs the Secretary of Commerce, acting through the Director of the National Institute of Standards and Technology (NIST), to develop a companion resource to the AI RMF, NIST AI 100 -1, for generative AI.

This work was informed by public feedback and consultations with diverse stakeholder groups as part of NIST's Generative AI Public Working Group (GAI PWG). The GAI PWG was an open, transparent, and collaborative process , facilitated via a virtual workspace, to obtain multistakeholder input on GAI risk management and to inform NIST's approach.

The focus of the GAI PWG was limited to four primary considerations relevant to GAI: Governance, Content Provenance, Pre -deployment Testing, and Incident Disclosure (further described in Appendix A). As such, the suggested actions in this document primarily address these considerations.

Future revisions of this profile will include additional AI RMF subcategories, risks, and suggested actions based on additional considerations of GAI as the space evolves and empirical evidence indicates additional risks . A glossary of terms pertinent to GAI risk management will be developed and hosted on NIST's Trustworthy &amp; Responsible AI Resource Center (AIRC), and added to The Language of Trustworthy AI: An In -Depth Glossary of Terms.

This document was also informed by public comments and consultations from several Request s for Information.

## 2. Overview of Risks Unique to or Exacerbated by GAI

In the context of the AI RMF, risk refers to the composite measure of an event's probability (or likelihood) of occurring and the magnitude or degree of the consequences of the corresponding event. Some risks can be assessed as likely to materialize in a given context, particularly those that have been empirically demonstrated in similar contexts. Other risks may be unlikely to materialize in a given context, o r may be more speculative and therefore uncertain.

AI risks can differ from or intensify traditional software risks. Likewise, GAI can exacerbate existing AI risks, and creates unique risks. GAI risks can vary along many dimensions:

- Stage of the AI lifecycle: Risks can arise during design , development , depl oyment , operation, and/or decommissioning.
- Scope: Risks may exist at individual model or system levels , at the application or implementation levels (i.e., for a specific use case), or at the ecosystem level -that is, beyond a single system or organizational context. Examples of the latter include the expansion of ' algorithmic monocultures , 3 ' resulting from repeated use of the same model, or impacts on access to opportunity, labor markets, and the creative economies . 4
- Source of risk: Risks may emerge from factors related to the de sign, training, or operation of the GAI model itself, stemming in some cases from GAI model or system inputs, and in other cases , from GAI system outputs. Many GAI risks, however, originate from human behavior , including

3 'Algorithmic monocultures' refers to the phenomenon in which repeated use of the same model or algorithm in consequential decision -making settings like employment and lending can result in increased susceptibility by systems to correlated failures (like unexpected shocks), due to multiple actors relying on the same algorithm.

4 Many studies have projected the impact of AI on the workforce and labor markets. Fewer studies have examined the impact of GAI on the labor market, though some industry surveys indicate that that both employees and employers are pondering this disruption .

- the abuse, misuse, and unsafe repurposing by humans (adversarial or not), and others result from interactions between a human and an AI system.
- Time scale: GAI risks may materialize abruptly or across extended periods . Example s include immediate (and/or prolonged) emotional harm and potential risks to physical safety due to the distribution of harmful deepfake images , or the lo ng-term effect of disinformation on soci etal trust in public institutions .

The presence of risks and where they fall along the dimensions above will vary depending on the characteristics of the GAI model, system, or use case at hand. These characteristics include but are not limited to GAI model or system architecture, training mechanisms and libraries , data types used for training or fine -tuning , levels of model access or availability of model weights, and application or use case context .

Organizations may choose to tailor how they measure GAI risks based on these characteristics . They may additionally wish to allocate risk management resources relative to the severity and likelihood of negative impact s , including where and how these risks manifest, and their direct and material impacts harms in the context of GAI use. Mitigations for model or system level risks may differ from mitigations for use-case or ecosystem level risks.

Importantly, some GAI risks are unknown, and are therefore difficult to properly scope or evaluate given the uncertainty about potential GAI scale, complexity, and capabilities. Other risks may be known but difficult to estimate given the wide range of GAI stakeholders, uses, inputs, and outputs . Challenges with risk estimation are aggravated by a lack of visibility into GAI training data, and the generally immature state of the science of AI measurement and safety today. This document focuses on risks for which there is an existing empirical evidence base at the time this profile was written ; for example, speculative risks that may potentially arise in more advanced, future GAI systems are not considered. Future updates may incorporate additional risks or provide further details on the risks identified below.

To guide organizations in identifying and managing GAI risks, a set of risks unique to or exacerbated by the development and use of GAI are defined below. 5 Each risk is labeled according to the outcome , object, or source of the risk (i.e., some are risks 'to' a subject or domain and others are risks 'of' or 'from' an issue or theme ). These risks provide a lens through which organizations can frame and execute risk management efforts. To help streamline risk management efforts, each risk is mapped in Section 3 (as well as in tables in Appendix B) to relevant Trustworthy AI Characteristics identified in the AI RMF .

5 These risks can be further categorized by organizations depending on their unique approaches to risk definition and management. One possible way to further categorize these risks, derived in part from the UK's International Scientific Report on the Safety of Advanced AI, could be: 1) Technical / Model risks (or risk from malfunction): Confabulation; Dangerous or Violent Recommendations; Data Privacy; Value Chain and Component Integration; Harmful Bias, and Homogenization ; 2) Misuse by humans (or malicious use): CBRN Information or Capabilities ; Data Privacy; HumanAI Configuration; Obscene, Degrading, and/or Abusive Content; Information Integrity; Information Security; 3) Ecosystem / societal risks (or systemic risks) : Data Privacy; Environmental; Intellectual Property . We also note that some risks are cross -cutting between these categories.

1. CBRN Information or Capabilities : E ased access to or synthesis of materially nefarious information or design capabilities related to chemical, biological, radiological, or nuclear (CBRN) weapons or other dangerous materials or agents.
2. Confabulation: The production of confidently stated but erroneous or false content (known colloquially as 'hallucinations' or 'fabrications') by which users may be misled or deceived. 6
3. Dangerous, Violent , or Hateful Content: Eased production of and access to violent, inciting, radicalizing, or threatening content as well as recommendations to carry out self -harm or conduct illegal activities. Includes d ifficulty controlling public exposure to hate ful and disparaging or stereotyping content.
4. Data Privacy: Impacts due to l eakage and unauthorized use , disclosure , or deanonymization of biometric, health, location, or other personally identifiable information or sensitive data . 7
5. Environmental Impacts: Impacts due to high compute resource utilization in training or operating GAI models, and related outcomes that may adversely impact ecosystems.
6. Harmful Bias or Homogenization: Amplification and exacerbation of historical , s ocietal, and systemic biases; performance disparities 8 between sub-groups or languages , possibly due to nonrepresentative training data , that result in discrimination, amplification of biases, or incorrect presumptions about performance ; undesired homogeneity that skews system or model outputs , which may be erroneous, lead to ill-founded decisionmaking, or amplify harmful biases.
7. Human -AI Configuration: Arrangements of or interactions between a human and an AI system which can result in the human inappropriately anthropomorphizing GAI system s or experiencing algorithmic aversion , automation bias, over-reliance , or emotional entanglement with GAI systems.
8. Information Integrity: Lowered barrier to entry to generate and support the exchange and consumption of content which may not distinguish fact from opinion or fiction or acknowledge uncertainties, or could be leveraged for large -scale dis- and misinformation campaigns.
9. Information Security: Lowered barriers for offensive cyber capabilities, including via automated discovery and exploitation of vulnerabilities to ease hacking, malware, phishing, offensive cyber

6  Some commenters have noted that the terms 'hallucination' and 'fabrication' anthropomorphize GAI, which itself is a risk related to GAI systems as it can inappropriately attribute human characteristics to non -human entities.

7 What is categorized as sensitive data or sensitive PII can be highly contextual based on the nature of the information, but examples of sensitive information include information that relates to an information subject's most intimate sphere, including political opinions, sex life, or criminal convictions .

8 The notion of harm presumes some baseline scenario that the harmful factor (e.g., a GAI model) makes worse . When the mechanism for potential harm is a disparity between groups, it can be difficult to establish what the most appropriate baseline is to compare against, which can result in divergent views on when a disparity between AI behaviors for different subgroups constitutes a harm. In discussing harms from disparities such as biased behavior, this document highlights examples where someone's situation is worsened relative to what it would have been in the absence of any AI system , making the outcome unambiguously a harm of the system.

- operations, or other cyberattacks ; increased attack surface for targeted cyberattacks, which may compromise a system's availability or the confidentiality or integrity of training data, code, or model weights.
10. Intellectual Property: Eased production or replication of alleged copyrighted, trademarked, or licensed content without authorization (possibly in situations which do not fall under fair use); eased exposure of trade secrets; or plagiarism or illegal replication .
11. Obscene, Degrading, and/or Abusive Content: Eased production of and access to obscene, degrading, and/or abusive imagery which can cause harm , including synthetic child sexual abuse material (CSAM), and nonconsensual intimate images (NCII) of adults.
12. Value Chain and Component Integration : Non-transparent or untraceable integration of upstream thirdparty components, including data that has been improperly obtained or not processed and cleaned due to increased automation from GAI; improper supplier vetting across the AI lifecycle ; or other issues that diminish transparency or accountability for downstream users.

## 2.1. CBRN Information or Capabilities

In the future, GAI may enable malicious actors to more easily access CBRN weapons and/or relevant knowledge, information, materials, tools, or technologies that could be misused to assist in the design, development, production, or use of CBRN weapons or other dangerous materials or agents. While relevant biological and chemical threat knowledge and information is often publicly accessible , LLMs could facilitate its analysis or synthesis , particularly by individuals without formal scientific training or expertise .

Recent research on this topic found that LLM outputs regarding biological threat creation and attack planning provided minimal assistance beyond traditional search engine queries, suggesting that state-ofthe-art LLMs at the time these studies were conducted do not substantially increase the operational likelihood of such an attack. The physical synthesis development, production, and use of chemical or biological agents will continue to require both applicable expertise and supporting materials and infrastructure. The impact of GAI on chemical or biological agent misuse will depend on what the key barriers for malicious actors are (e.g., whether information access is one such barrier ), and how well GAI can help actors address those barriers.

Furthermore , chemical and biological design tools (BDTs) - highly specialized AI systems trained on scientific data that aid in chemical and biological design - may augment design capabilities in chemistry and biology beyond what text -based LLMs are able to provide . As these models become more efficacious, including for beneficial uses, it will be important to assess their potential to be used for harm, such as the ideation and design of novel harmful chemical or biological agents.

While some of these described capabilities lie beyond the reach of existing GAI tools, ongoing assessments of this risk would be enhanced by monitoring both the ability of AI tools to facilitate CBRN weapons planning and GAI systems' connection or access to relevant data and tools.

Trustworthy AI Characteristic

: Safe , Explainable and Interpretable

## 2.2. Confabulation

'Confabulation' refers to a phenomenon in which GAI systems generate and confidently present erroneous or false content in response to prompts. Confabulations also include generated outputs that diverge from the prompts or other input or that contradict previously generated statements in the same context. Th ese phenomena are colloquially also referred to as 'hallucination s' or ' fabrication s.'

Confabulations can occur across GAI outputs and contexts . 9, 10 Confabulations are a natural result of the way generative models are designed: they generate outputs that approximate the statistical distribution of their training data; f or example, LLMs predict the next t oken or word in a sentence or phrase. While such statistical prediction can produce factually accurate and consistent outputs , it can also produce outputs that are factually inaccurate or internally inconsistent. This dynamic is particularly relevant when it comes to open-ended prompts for long-form responses and in domains which require highly contextual and/or domain expertise.

Risks from confabulations may arise when users believe false content -often due to the confident nature of the response - leading users to act upon or promote the false information. This poses a challenge for many realworld applications, such as in healthcare, where a confabulated summary of patient information reports could cause doctors to make incorrect diagnoses and/or recommend the wrong treatments. Risks of confabulated content may be especially important to monitor when integrating GAI into applications involving consequential decision making .

GAI outputs may also include confabulated logic or citations that purport to justify or explain the system's answer, which may further mislead humans into inappropriately trusting the system's output . For instance, LLMs sometimes provide logical steps for how they arrived at an answer even when the answer itself is incorrect. Similarly, an LLM could falsely assert that it is human or has human traits, potentially deceiving humans into believing they are speaking with another human .

T he extent to which humans can be deceived by LLMs, the mechanisms by which this may occur, and the potential risks from adversarial prompting of such behavior are emerging areas of study. Given the wide range of downstream impacts of GAI, it is difficult to estimate the downstream scale and impact of confabulations .

Trustworthy AI Characteristics: Fair with Harmful Bias Managed, Safe, Valid and Reliable, Explainable and Interpretable

## 2.3. Dangerous, Violent , or Hateful Content

GAI systems can produce content that is inciting, radicalizing, or threatening, or that glorifi es violence , with greater ease and scale than other technologies . LLMs have been reported to generate dangerous or violent recommendations, and some models have generated actionable instructions for dangerous or

9 Confabulations of falsehoods are most commonly a problem for text -based outputs; for audio, image, or video content, creative generation of non -factual content can be a desired behavior.

10 For example, legal confabulations have been shown to be pervasive in current state-of-theart LLMs. See also, e.g.,