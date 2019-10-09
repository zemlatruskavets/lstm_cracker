1) read in data
    1) clean up data (duplicates, NaN, etc)
2) get data characteristics
    1) determine number of characters
    2) determine/define longest sequence length
3) generator
    3.1) tokenization
    3.2) sliding windows
4) training
5) sequence
    5.1) for i in sequence, predict most likely candidates in each position
    5.2) calculate most likely shared candidates
    5.3) calculate probabilities of overall adjusted sequences

