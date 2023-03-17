s = 'aaaabbboooaooooobooo'
n = 0
mi = 0
start = 'a'
end = 'b'
for i in range(len(s)):
    if i != 0 and s[i] == 'o' and s[i-1] != 'o': 
        start = s[i-1]
        n += 1
    elif i != 0 and i != len(s) - 1 and s[i] == 'o':   n += 1
    elif i != 0 and s[i] != 'o' and s[i-1] == 'o' :   
        end = s[i]
        print(i, n, mi)
        if start != end:
            if n < mi or mi == 0:
                mi = n
            n = 0
print(mi)