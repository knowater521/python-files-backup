for a in ['x','y','z']:
    for b in ['x', 'y', 'z']:
        for c in ['x', 'y', 'z']:
            if(a!=b)and(b!=c)and(c!=a) and (a!='x') and (c!='x') and (c!='z'):
                print('a<->%s，b<->%s，c<->%s' %(a,b,c))