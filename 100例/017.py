str=input('Please enter a string:')
letters=0
space=0
digit=0
others=0
for c in str:
	if 'a'<=c<='z' or 'A'<=c<='Z':  #也可以导入String库，分别用c.isalpha(),c.isdigit(),c.isspace()
		letters+=1
	elif c==' ':
		space+=1
	elif '0'<=c<='9':
		digit+=1
	else:
		others+=1
		
print('letters=%d,space=%d,digit=%d,others=%d'%(letters,space,digit,others))
