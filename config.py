import torch
import os

# ── Paths ──────────────────────────────────────────────
MODEL_PATH       = "model/bank_doc_classifier_multimodal.pth"
DOCS_FOLDER      = "stored_documents"
UPLOAD_TEMP      = "uploads_temp"
QUEUE_UPLOAD_DIR = "queue_uploads"

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_TEMP, exist_ok=True)
os.makedirs(QUEUE_UPLOAD_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Ollama ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_CHAT  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
QWEN_MODEL   = "qwen2.5vl:3b"

# Timeouts streaming (secondes sans recevoir de token)
# llama3.2 : texte → rapide
OLLAMA_TIMEOUT     = 120   # timeout_no_token pour classification
OLLAMA_MAX_RETRIES = 1

# Qwen2.5-VL : vision → plus lent, surtout sans GPU
# Augmenter à 90-120s si le modèle est très lent sur votre machine
QWEN_TIMEOUT = 120

# ── MongoDB ────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB  = "bankdoc"

# ── Pipeline ───────────────────────────────────────────
TRAINING_THRESHOLD = 50

# Seuils de confiance (utilisés pour affichage UI uniquement dans le nouveau pipeline)
# La décision passe maintenant par la double confirmation LLM, pas par ces seuils.
CONF_HIGH   = 0.80
CONF_MEDIUM = 0.60

# Seuil OCR
OCR_QUALITY_THRESHOLD = 0.30
OCR_MIN_LENGTH        = 200   # chars minimum avant re-OCR

# ── Keyword rules (consultatif — non décisionnel dans le pipeline) ─────
KEYWORD_RULES = {
    "acte_heredite": [
        "acte de notoriété", "succession", "héritier", "héritière", "défunt",
        "dévolution successorale", "notaire", "héritage", "ayant droit",
        "décès", "hoirie", "masse successorale",
    ],
    "rib": [
        "IBAN", "BIC", "RIB", "domiciliation bancaire", "titulaire du compte",
        "code banque", "code guichet", "numéro de compte", "clé RIB",
    ],
    "cheque": [
        "chèque", "payez contre ce chèque", "à l'ordre de", "chéquier",
        "tiré sur", "montant en lettres", "endossement", "barrement",
    ],
    "tableau_amortissement": [
        "tableau d'amortissement", "amortissement", "échéance", "mensualité",
        "capital restant dû", "intérêts", "taux d'intérêt", "durée du prêt",
        "remboursement", "annuité", "période", "capital amorti",
    ],
    "acte_naissance": [
        "acte de naissance", "né(e) le", "commune de naissance", "extrait",
        "officier d'état civil", "naissance", "registre des naissances",
        "filiation", "père", "mère", "lieu de naissance",
    ],
    "assurance": [
        "police d'assurance", "assuré", "assureur", "prime", "garantie",
        "sinistre", "contrat d'assurance", "couverture", "franchise",
        "bénéficiaire", "risque assuré", "cotisation", "résiliation",
    ],
    "attestation_solde": [
        "attestation de solde", "solde créditeur", "solde du compte",
        "certifions", "attestons", "arrêté au", "situation de compte",
        "disponibilité", "avoir en compte", "banque atteste",
    ],
    "carte_identite": [
        "carte nationale d'identité", "cni", "cin", "carte d'identité",
        "né le", "n° cin", "numéro cin", "national identity card",
        "بطاقة الوطنية", "identity card", "pièce d'identité",
    ],
}