import numpy as np 

functions = [
    [
        lambda bid : 60*(np.power(2*bid,2))/np.power(np.power(2*bid,4) + 1, 1/2),
        lambda bid : 150*(1-1/(np.cosh(5.7*bid))),
        lambda bid : 100*(1-np.exp(-5*bid))
    ],
    [
        lambda bid : 80*np.tanh(9*bid),
        lambda bid : 120*(1-np.exp(-np.power(5/2*bid, 5/3))),
        lambda bid : 100*(np.power(2*bid,2))/np.power(np.power(4*bid,4) + 20, 1/2)
    ],
    [
        lambda bid : 100*(1-1/(np.cosh(6.3*bid))),
        lambda bid : 50*(1-np.cosh(1)/(np.cosh(5.7*bid + 1))),
        lambda bid : 120*np.tanh(12*bid)
    ],
    [
        lambda bid : 70*(1-np.exp(-np.power(3*bid,3/2))),
        lambda bid : 50*(np.power(3*bid,5/2))/np.power(np.power(2*bid,11/2) + 7, 1/2),
        lambda bid : 100*np.tanh(4*bid)
    ],
    [
        lambda bid : 80*(1-np.exp(-4.5*bid)),
        lambda bid : 40*(1-np.exp(-18*bid)),
        lambda bid : 80*(1-np.exp(-np.power(3*bid,3/2)))
    ]
]

functions_tuple = [
    [
        ( 1, lambda bid : 60*(np.power(2*bid,2))/np.power(np.power(2*bid,4) + 1, 1/2) ),
        ( 2, lambda bid : 150*(1-1/(np.cosh(5.7*bid))) ),
        ( 3, lambda bid : 100*(1-np.exp(-5*bid)) )
    ],
    [
        ( 4, lambda bid : 80*np.tanh(9*bid) ),
        ( 5, lambda bid : 120*(1-np.exp(-np.power(5/2*bid, 5/3))) ),
        ( 6, lambda bid : 100*(np.power(2*bid,2))/np.power(np.power(4*bid,4) + 20, 1/2) )
    ],
    [
        ( 7, lambda bid : 100*(1-1/(np.cosh(6.3*bid))) ),
        ( 8, lambda bid : 50*(1-np.cosh(1)/(np.cosh(5.7*bid + 1))) ),
        ( 9, lambda bid : 120*np.tanh(12*bid) ) 
    ],
    [
        ( 10, lambda bid : 70*(1-np.exp(-np.power(3*bid,3/2))) ),
        ( 11, lambda bid : 50*(np.power(3*bid,5/2))/np.power(np.power(2*bid,11/2) + 7, 1/2) ),
        ( 12, lambda bid : 100*np.tanh(4*bid) )
    ],
    [
        ( 13, lambda bid : 80*(1-np.exp(-4.5*bid)) ),
        ( 14, lambda bid : 40*(1-np.exp(-18*bid)) ),
        ( 15, lambda bid : 80*(1-np.exp(-np.power(3*bid,3/2))) )
    ]
]

def get_functions():
    return functions