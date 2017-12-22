score=int(input('Please input the score:'))
if score >= 90:
    grade = 'A'
elif score<60:
    grade = 'C'
else:
    grade = 'B'
print('The grade of %d is:'%score,grade)
