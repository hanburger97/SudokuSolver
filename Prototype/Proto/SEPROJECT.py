from GoogleVision import google_vision
from camera import takePhoto
import os
import subprocess

def nth_repl(s, sub, repl, nth):
    find = s.find(sub)
    # if find is not p1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != nth:
        # find + 1 means we start at the last match start index + 1
        find = s.find(sub, find + 1)
        i += 1
    # if i  is equal to nth we found nth matches so replace
    if i == nth:
        return s[:find]+repl+s[find + len(sub):]
    return s

#filters
alts = {}
alts[' include '] = ['#include ']
alts[')\n'] = [');','){']
alts['|'] = [')','(']
alts['\n'] = [';']
#this one increases processing time a ton
#alts['l'] = [')','(']
alts['Stlio.h'] = ['<stdio.h']
#filter outputs
codeSamples = []

#function to take text and apply the above filters
#adds every possible combination of the above filters to codeSamples
def recursiveFunc(text):
    if(not text in codeSamples):
        #print(text)
        codeSamples.append(text)
    for original, corrections in alts.items():
        for i in range(0,text.count(original)):
            for correction in corrections:
                s = nth_repl(text, original, correction, i+1)                
                recursiveFunc(s)

              
filename = "input.jpg"

#takes photo
takePhoto(filename)

#some initial things pre filter
text = google_vision(filename)
text = " " + text
if(not text[len(text)-1]=="}"):
    text = text+"}"


recursiveFunc(text)

codeOutput = ""

#tries to run the codeSamples, breaks once one runs
for s in codeSamples:
    if(s.count("{")==s.count("}") and s.count("(")==s.count(")")):
        text_file = open("Output.c", "w")
        text_file.write(s)
        text_file.close()

        cmd = [ 'gcc', 'Output.c']
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE ).communicate()[0]
        output = output.decode()

        if(len(output)<=0):
            cmd = [ './a.out']
            output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
            output = output.decode()
            codeOutput = output
            break
        
        if(filteredCodeSamples.index(s)==len(filteredCodeSamples)-1):
            codeOutput = "Could not run, please try taking another photo."

print("OUTPUT: "+codeOutput)    


    






