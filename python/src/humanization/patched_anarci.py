from anarci.anarci import _domains_are_same, _hmm_alignment_to_states


def _parse_hmmer_query(query, bit_score_threshold=80, hmmer_species=None):
    """

    @param query: hmmer query object from Biopython
    @param bit_score_threshold: the threshold for which to consider a hit a hit.

    The function will identify multiple domains if they have been found and provide the details for the best alignment for each domain.
    This allows the ability to identify single chain fvs and engineered antibody sequences as well as the capability in the future for identifying constant domains.

    """
    hit_table = [['id', 'description', 'evalue', 'bitscore', 'bias',
                  'query_start', 'query_end']]

    # Find the best hit for each domain in the sequence.

    top_descriptions, domains, state_vectors = [], [], []

    if query.hsps:  # We have some hits
        # If we have specified a species, check to see we have hits for that species
        # Otherwise revert back to using any species
        if hmmer_species:
            # hit_correct_species = [hsp for hsp in query.hsps if hsp.hit_id.startswith(hmmer_species) and hsp.bitscore >= bit_score_threshold]
            hit_correct_species = []
            for hsp in query.hsps:
                if hsp.bitscore >= bit_score_threshold:
                    for species in hmmer_species:
                        if hsp.hit_id.startswith(species):
                            hit_correct_species.append(hsp)

            if hit_correct_species:
                hsp_list = hit_correct_species
            else:
                print("Drop limiting, empty result")
                hsp_list = hit_correct_species
        else:
            hsp_list = query.hsps

        for hsp in sorted(hsp_list, key=lambda x: x.evalue):  # Iterate over the matches of the domains in order of their e-value (most significant first)
            new = True
            if hsp.bitscore >= bit_score_threshold:  # Only look at those with hits that are over the threshold bit-score.
                for i in range(len(domains)):  # Check to see if we already have seen the domain
                    if _domains_are_same(domains[i], hsp):
                        new = False
                        break
                hit_table.append([hsp.hit_id, hsp.hit_description, hsp.evalue, hsp.bitscore, hsp.bias, hsp.query_start,
                                  hsp.query_end])
                if new:  # It is a new domain and this is the best hit. Add it for further processing.
                    domains.append(hsp)
                    top_descriptions.append(
                        dict(list(zip(hit_table[0], hit_table[-1]))))  # Add the last added to the descriptions list.

        # Reorder the domains according to the order they appear in the sequence.
        ordering = sorted(list(range(len(domains))), key=lambda x: domains[x].query_start)
        domains = [domains[_] for _ in ordering]
        top_descriptions = [top_descriptions[_] for _ in ordering]

    ndomains = len(domains)
    for i in range(ndomains):  # If any significant hits were identified parse and align them to the reference state.
        domains[i].order = i
        species, chain = top_descriptions[i]["id"].split("_")
        state_vectors.append(
            _hmm_alignment_to_states(domains[i], ndomains, query.seq_len))  # Alignment to the reference states.
        top_descriptions[i]["species"] = species  # Reparse
        top_descriptions[i]["chain_type"] = chain
        top_descriptions[i]["query_start"] = state_vectors[-1][0][
            -1]  # Make sure the query_start agree if it was changed

    return hit_table, state_vectors, top_descriptions
