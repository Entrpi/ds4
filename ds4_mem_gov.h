/* ds4_mem_gov.h -- memgov D0b: shadow governor core (pure, leaf).
 *
 * Lease, claim, ledger and quote types plus the checked requester-aware
 * evaluator from the governance design (deepmem plan sec 6.2).  D0b runs
 * this ENTIRELY IN SHADOW: the live formulas keep deciding, the evaluator
 * is called beside them and its quotes are only counted and logged.
 * Enforcement is D2's.
 *
 * Everything here is pure: the evaluator maps (ledger image, observation,
 * claim) -> quote with no globals, no clock, no allocation.  That purity
 * IS the fake memory provider -- tests construct inputs directly and
 * cover the arithmetic and state machine without a GPU or a hook.
 *
 * LEAF CONTRACT: no ds4.h, no backend headers.  ds4_mem_census.h is
 * itself a leaf (types + checked arithmetic) and is the one include.
 * Both TUs (ds4_server.c, ds4_cuda.cu) and the unit suite share these
 * shapes by value. */
#ifndef DS4_MEM_GOV_H
#define DS4_MEM_GOV_H

#include <stdint.h>
#include "ds4_mem_census.h"

/* =========================================================================
 * Consumers (plan sec 6.1, narrowed to what exists on this tree; the
 * scoping table local/docs/v057/d0b_scoping_2026-08-11.md sec 2 maps each
 * to its census classes for the reconciliation gate).  FIXED cardinality:
 * growing this enum is a commit-visible act. */
typedef enum {
    DS4_GOVC_ENGINE_BOOT = 0,   /* weights + KV + mirrors + staging       */
    DS4_GOVC_PREWARM,           /* boot prewarm scratch growth            */
    DS4_GOVC_BATCH_BANK_PLAN,   /* bank comp/index demand-mapped growth   */
    DS4_GOVC_SERIAL_SESSION,    /* serial lazy graph incl. right-size;
                                   the opt-in serial reserve is THIS
                                   lease's reservation field -- one lane,
                                   one owner, so the sec 6.2 asymmetry
                                   (serial spends its own reserve, batch
                                   must not) falls out of ownership       */
    DS4_GOVC_STATIC_BATCH,      /* per-call /v1/batch fallback graph      */
    DS4_GOVC_VISION_ENCODER,    /* Track B inc 9: the Vision-Exp encoder's
                                   per-encode device scratch (transient,
                                   published as the lease's intent for the
                                   duration of one encode)                */
    DS4_GOVC__COUNT
} ds4_gov_consumer;

/* Absolute published state per consumer (sec 6.2: leases replace, never
 * accumulate -- re-publishing the same state is idempotent by design).
 *
 *   intent      absolute committed intent in bytes, INCLUDING credited-
 *               but-unfaulted growth (the cont page-union projection is
 *               one such absolute: it already carries every credit and
 *               the candidate, so it is published as-is, never summed
 *               with its parts);
 *   resident    the physically backed portion of intent (what live free
 *               already cannot see);
 *   reservation carve this consumer's lane holds against OTHER
 *               requesters (the serial reserve).  Protected for others,
 *               spendable by the owner (sec 6.2 requester-aware rule).
 *
 * Invariant: resident <= intent (checked by the evaluator; violation is
 * an impossible state and fails closed). */
typedef struct {
    uint64_t intent;
    uint64_t resident;
    uint64_t reservation;
} ds4_gov_lease;

/* Immutable ledger image the evaluator consumes.  D0b-2 wraps the live
 * instance in the versioned epoch protocol; tests and D0b-1 construct
 * plain images.  floor/substrate ride in the image so evaluation is a
 * pure function of its arguments. */
typedef struct {
    ds4_gov_lease lease[DS4_GOVC__COUNT];
    uint64_t floor_bytes;            /* operator floor (--mem-floor-gb)   */
    uint64_t substrate_outstanding;  /* deferred substrate charge         */
    uint64_t epoch;                  /* image stamp (D0b-2 protocol)      */
} ds4_gov_ledger;

/* One admission/growth ask.  proposed_outstanding is the ABSOLUTE
 * replacement for the requester's lease intent -- not a delta (sec 6.2:
 * prospective = all - claimant_old + claimant_proposed).  class_limit is
 * the cap this claim evaluates under (0 = no class cap); in shadow mode
 * the live code still owns cap policy, so the claim carries it. */
typedef struct {
    int requester;                   /* ds4_gov_consumer                  */
    int memc;                        /* ds4_mem_class the bytes land in   */
    int domain;                      /* ds4_mem_domain                    */
    uint64_t proposed_outstanding;   /* absolute intent replacement       */
    uint64_t operation_transient;    /* peak transient during the op      */
    uint64_t class_limit;            /* cap for class verdict; 0 = none   */
} ds4_gov_claim;

/* Quote status (sec 6.2: the decision returns the complete quote, not a
 * boolean).  Retryability convention:
 *   REFUSE_CLASS  not retryable -- a boot-plan cap does not move;
 *   REFUSE_LIVE   retryable     -- trim/decay can fund the deficit;
 *   RETRY_OBS     retryable     -- transient query error (sec 6.4: new
 *                 allocations fail closed or retry; existing work holds
 *                 backing and continues);
 *   UNSUPPORTED   not retryable -- CPU/Metal documented policy, the
 *                 shadow records and never acts;
 *   FAULT         not retryable -- overflow / inconsistent ownership /
 *                 impossible state fails closed (caller counts it). */
typedef enum {
    DS4_GOV_ADMIT = 0,
    DS4_GOV_REFUSE_CLASS,
    DS4_GOV_REFUSE_LIVE,
    DS4_GOV_RETRY_OBS,
    DS4_GOV_UNSUPPORTED,
    DS4_GOV_FAULT,
    DS4_GOV_STATUS__COUNT
} ds4_gov_status;

/* D0b-3: old/new verdict comparison, a CLOSED reason set (plan sec 12
 * D0b: "expose old/new verdict disagreements with fixed reasons").
 * Growing this enum is a commit-visible act; there is no OTHER bucket by
 * design -- an unclassifiable pair is a FAULT, never a shrug. */
typedef enum {
    DS4_GOV_CMP_AGREE = 0,          /* same verdict class                 */
    DS4_GOV_CMP_LIVE_STRICTER,      /* live refused, shadow admits        */
    DS4_GOV_CMP_SHADOW_STRICTER,    /* live proceeded, shadow refuses     */
    DS4_GOV_CMP_VERDICT_CLASS,      /* both refuse, different refusal     */
    DS4_GOV_CMP_OBS_POLICY,         /* shadow has no OK observation where
                                       the live formula failed open       */
    DS4_GOV_CMP_FAULT,              /* evaluator fault -- no comparison   */
    DS4_GOV_CMP__COUNT
} ds4_gov_cmp;

/* Classify a (live verdict, shadow quote status) pair.  live_status is
 * the LIVE formula's outcome mapped by the call site onto the quote
 * vocabulary (ADMIT / REFUSE_CLASS / REFUSE_LIVE -- the only three the
 * live formulas can express). */
static inline int ds4_gov_compare(int live_status, int shadow_status) {
    if (shadow_status == DS4_GOV_FAULT) return DS4_GOV_CMP_FAULT;
    if (shadow_status == DS4_GOV_RETRY_OBS ||
        shadow_status == DS4_GOV_UNSUPPORTED)
        return DS4_GOV_CMP_OBS_POLICY;
    if (live_status == shadow_status) return DS4_GOV_CMP_AGREE;
    if (live_status == DS4_GOV_ADMIT) return DS4_GOV_CMP_SHADOW_STRICTER;
    if (shadow_status == DS4_GOV_ADMIT) return DS4_GOV_CMP_LIVE_STRICTER;
    return DS4_GOV_CMP_VERDICT_CLASS;   /* both refusals, different class */
}

/* =========================================================================
 * memgov D2-1: per-consumer governance mode (plan sec 12 D2).
 *
 * The TEMPORARY migration vocabulary: every consumer family runs under
 * off | observe | enforce until D4 deletes the legacy formulas and this
 * mode with them.  Parsed ONCE at engine open into a process mode table
 * (no hot-path getenv); the table's accessor defaults to OBSERVE before
 * init so a missed init can never silently disable the shadow.
 *
 *   OFF      kill switch: no evaluation, no publication -- zero governor
 *            activity (snapshots read the empty ledger, documented);
 *   OBSERVE  the D0b contract: live decides, quotes counted/disclosed;
 *   ENFORCE  the governor's quote IS the verdict; the legacy formula
 *            keeps being computed as the comparison target, so the
 *            memgov_decisions matrix stays the oracle in both
 *            directions (healthy legs assert zero *_STRICTER exactly
 *            as observe legs do). */
typedef enum {
    DS4_GOV_MODE_OFF = 0,
    DS4_GOV_MODE_OBSERVE,
    DS4_GOV_MODE_ENFORCE,
    DS4_GOV_MODE__COUNT
} ds4_gov_mode_t;

/* Pure vocabulary helpers (unit-tested on CPU).  parse returns -1 on an
 * unrecognized value -- the caller decides loudness; it never guesses. */
static inline int ds4_gov_mode_parse(const char *s) {
    if (!s) return -1;
    if (s[0] == 'o' && s[1] == 'f' && s[2] == 'f' && !s[3])
        return DS4_GOV_MODE_OFF;
    if (s[0] == 'o' && s[1] == 'b') {                 /* observe */
        const char *t = "observe";
        for (int i = 0; ; i++) {
            if (t[i] != s[i]) return -1;
            if (!t[i]) return DS4_GOV_MODE_OBSERVE;
        }
    }
    if (s[0] == 'e' && s[1] == 'n') {                 /* enforce */
        const char *t = "enforce";
        for (int i = 0; ; i++) {
            if (t[i] != s[i]) return -1;
            if (!t[i]) return DS4_GOV_MODE_ENFORCE;
        }
    }
    return -1;
}

static inline const char *ds4_gov_mode_name(int m) {
    switch (m) {
    case DS4_GOV_MODE_OFF:     return "off";
    case DS4_GOV_MODE_OBSERVE: return "observe";
    case DS4_GOV_MODE_ENFORCE: return "enforce";
    default:                   return "?";
    }
}

/* The complete decision record (sec 6.2 list, field for field). */
typedef struct {
    int status;                      /* ds4_gov_status                    */
    int retryable;
    int requester, memc, domain;
    /* raw observation */
    int obs_status, obs_source;      /* ds4_mem_obs_status / _source      */
    uint64_t obs_free, obs_total;
    /* class verdict inputs */
    uint64_t class_limit, proposed_class_bytes;
    /* live verdict inputs */
    uint64_t floor_bytes;
    uint64_t other_reservations;     /* sum of reservations NOT owned by
                                        the requester                     */
    uint64_t substrate_outstanding;
    uint64_t old_intent, proposed_intent, total_prospective_intent;
    uint64_t operation_transient;
    /* live verdict arithmetic */
    uint64_t available;              /* = obs_free                        */
    uint64_t required;               /* protected + unfunded spend        */
    uint64_t deficit;                /* required - available when short   */
    uint64_t epoch;                  /* ledger image stamp                */
} ds4_gov_quote;

/* Absolute lease publication with invariant checks (state-machine tests
 * and the D0b-3 wiring both use this; the live instance is written only
 * under the D0b-2 epoch brackets).  Publishing resident > intent is the
 * impossible state: the fault is counted and the publish REFUSED so the
 * ledger never holds a state the evaluator must fail on. */
static inline int ds4_gov_lease_publish(ds4_gov_ledger *lg, int consumer,
                                        uint64_t intent, uint64_t resident,
                                        uint64_t reservation,
                                        uint64_t *faults) {
    if (!lg || consumer < 0 || consumer >= DS4_GOVC__COUNT ||
        resident > intent) {
        if (faults) (*faults)++;
        return 0;
    }
    lg->lease[consumer].intent = intent;
    lg->lease[consumer].resident = resident;
    lg->lease[consumer].reservation = reservation;
    return 1;
}

/* The checked requester-aware evaluator (sec 6.2):
 *
 *   class_ok = proposed_class_reserved <= class_limit
 *   protected = floor
 *             + reservations_not_owned_by(requester)
 *             + substrate_outstanding
 *             + unfunded spend (proposed_intent - resident)
 *             + requester_operation_transient
 *   live_ok  = raw observed free >= protected
 *
 * Verdict order mirrors the live admission block (class cap first, live
 * floor second) so shadow/live comparisons are per-verdict meaningful.
 * Any checked-arithmetic saturation or impossible state yields FAULT --
 * never a wrapped number admitted as plausible.  The quote is filled as
 * a complete record on EVERY path, including refusals and faults. */
static inline ds4_gov_quote ds4_gov_evaluate(const ds4_gov_ledger *lg,
                                             const ds4_mem_observation *obs,
                                             const ds4_gov_claim *cl) {
    ds4_gov_quote q = {0};           /* complete record on every path     */
    uint64_t ovf = 0;                /* local checked-arithmetic faults   */
    if (!lg || !obs || !cl ||
        cl->requester < 0 || cl->requester >= DS4_GOVC__COUNT ||
        cl->memc < 0 || cl->memc >= DS4_MEMC__COUNT ||
        cl->domain < 0 || cl->domain >= DS4_MEMD__COUNT) {
        q.status = DS4_GOV_FAULT;
        return q;
    }
    q.requester = cl->requester;
    q.memc = cl->memc;
    q.domain = cl->domain;
    q.epoch = lg->epoch;
    q.obs_status = obs->status;
    q.obs_source = obs->source;
    q.class_limit = cl->class_limit;
    q.proposed_class_bytes = cl->proposed_outstanding;
    q.floor_bytes = lg->floor_bytes;
    q.substrate_outstanding = lg->substrate_outstanding;
    q.operation_transient = cl->operation_transient;
    q.old_intent = lg->lease[cl->requester].intent;
    q.proposed_intent = cl->proposed_outstanding;

    /* Requester-aware protected sums (sec 6.2).  "Outstanding" in the
     * plan's live gate is promised-but-unallocated debt (sec 6.3 reduces
     * it as bytes become physical), i.e. intent - resident per lease --
     * resident bytes are already invisible to raw free and must not be
     * double-counted.  prospective = all unfunded, with the claimant's
     * term replaced by its proposed unfunded spend.  A lease with
     * resident > intent ANYWHERE in the image is an impossible state:
     * fail closed rather than gate against a corrupt ledger. */
    uint64_t prospective_unfunded = 0;
    for (int c = 0; c < DS4_GOVC__COUNT; c++) {
        const ds4_gov_lease *l = &lg->lease[c];
        if (l->resident > l->intent) {
            q.status = DS4_GOV_FAULT;
            return q;
        }
        if (c != cl->requester) {
            q.other_reservations =
                ds4_mem_sat_add(q.other_reservations, l->reservation, &ovf);
            prospective_unfunded =
                ds4_mem_sat_add(prospective_unfunded,
                                l->intent - l->resident, &ovf);
        }
        q.total_prospective_intent =
            ds4_mem_sat_add(q.total_prospective_intent,
                            c == cl->requester ? cl->proposed_outstanding
                                               : l->intent, &ovf);
    }
    if (ovf) {                       /* saturated sums cannot quote        */
        q.status = DS4_GOV_FAULT;
        return q;
    }
    /* Observation policy (sec 6.4) before the verdicts: without an OK
     * answer there is nothing to gate against. */
    if (obs->status == DS4_MEMOBS_UNSUPPORTED) {
        q.status = DS4_GOV_UNSUPPORTED;
        return q;
    }
    if (obs->status != DS4_MEMOBS_OK) {
        q.status = DS4_GOV_RETRY_OBS;
        q.retryable = 1;
        return q;
    }
    q.obs_free = obs->free_bytes;
    q.obs_total = obs->total_bytes;
    q.available = obs->free_bytes;
    /* Class verdict (cap 0 = uncapped). */
    if (cl->class_limit != 0 &&
        cl->proposed_outstanding > cl->class_limit) {
        q.status = DS4_GOV_REFUSE_CLASS;
        return q;
    }
    /* Live verdict.  The claimant's term: a shrink (proposed <= resident)
     * spends nothing. */
    const uint64_t resident = lg->lease[cl->requester].resident;
    const uint64_t unfunded =
        cl->proposed_outstanding > resident
            ? cl->proposed_outstanding - resident : 0;
    uint64_t req = ds4_mem_sat_add(lg->floor_bytes,
                                   lg->substrate_outstanding, &ovf);
    req = ds4_mem_sat_add(req, q.other_reservations, &ovf);
    req = ds4_mem_sat_add(req, prospective_unfunded, &ovf);
    req = ds4_mem_sat_add(req, unfunded, &ovf);
    req = ds4_mem_sat_add(req, cl->operation_transient, &ovf);
    q.required = req;
    if (ovf) {                       /* saturated inputs cannot decide    */
        q.status = DS4_GOV_FAULT;
        return q;
    }
    if (q.available >= q.required) {
        q.status = DS4_GOV_ADMIT;
        return q;
    }
    q.status = DS4_GOV_REFUSE_LIVE;
    q.retryable = 1;
    q.deficit = q.required - q.available;
    return q;
}

/* =========================================================================
 * memgov D0b-2: versioned snapshot protocol (plan sec 6.3).
 *
 * A seqlock over a single-writer ledger: the writer marks the epoch odd,
 * mutates, publishes, marks it even; a reader samples epoch -> data ->
 * epoch and discards the copy unless both samples match and are even.
 * The reader's data copy itself may tear WHILE it is being discarded --
 * that benign race is the protocol (the C-standard caveat is documented
 * at the instantiation sites, as for the D0a copy loop).
 *
 * Single-writer is a PRECONDITION (the census writers all run under the
 * server's gen_mu or pre-listener boot), but it is TRIPWIRED here, not
 * assumed (plan sec 6.3 last para): write_begin faults on an odd epoch
 * (someone else mid-flight), write_end faults on an even one (unbalanced
 * bracket).  Both still increment, so parity self-heals after any
 * balanced sequence and readers keep discarding through the violation
 * window -- the tripwire is a loud counter, never a lock.
 *
 * SEQ_CST fences: these run at durable allocation points only (never
 * per-token), so the strongest fence is free and unambiguous. */
static inline void ds4_gov_epoch_write_begin(uint64_t *epoch,
                                             uint64_t *faults) {
    const uint64_t s = __atomic_load_n(epoch, __ATOMIC_RELAXED);
    if (s & 1u) { if (faults) (*faults)++; }   /* single-writer violated */
    __atomic_store_n(epoch, s + 1u, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline void ds4_gov_epoch_write_end(uint64_t *epoch,
                                           uint64_t *faults) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    const uint64_t s = __atomic_load_n(epoch, __ATOMIC_RELAXED);
    if (!(s & 1u)) { if (faults) (*faults)++; } /* unbalanced bracket */
    __atomic_store_n(epoch, s + 1u, __ATOMIC_RELAXED);
}

static inline uint64_t ds4_gov_epoch_read_begin(const uint64_t *epoch) {
    return __atomic_load_n(epoch, __ATOMIC_ACQUIRE);
}

/* 1 = the copy taken since read_begin is coherent (same epoch, even). */
static inline int ds4_gov_epoch_read_verify(const uint64_t *epoch,
                                            uint64_t began) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    const uint64_t now = __atomic_load_n(epoch, __ATOMIC_RELAXED);
    return now == began && !(began & 1u);
}

#endif /* DS4_MEM_GOV_H */
