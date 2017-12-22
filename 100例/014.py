n=int(input('Please input the number:'))

print(n,'=')
while n not in [1]:   #这个技巧不错，n一直减小直至为0，故可一直循环至n=0。
        for i in range(2,n+1):
                if n%i  == 0:
                        n//=i
                        if n == 1:
                                print(i)
                        else:
                                print(n,i,'*',end=' ')
                        break
