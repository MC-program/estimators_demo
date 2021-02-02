#import xml.etree.ElementTree as ET
import xmltodict




def read(filename):

    source       = []   ## independent vars  
    target       = []   ## target var
    content = ""


    try:
        content = readFile(filename)
        """
        tree    = ET.fromstring(content)   ### list
        for vector in tree:
            for v in vector:    # "coordinate"=domain, "energy"=target
                print(v)
            break
        """
    except:
        pass

    #dictRaw = ET.parse(content)
    dictRaw = xmltodict.parse(content)

    for key,vectors in dictRaw["data"].items():
        #print(vectors)
        #print(dictRaw["data"][vectors])
        for k,v in vectors.items():
            if k == "energy":          ### target value
                target.append(float(v))
            elif k == "coordinates":
                source.append([])   # <-- all coordinates
                for coord in v["c"]:      # ordered dictionary with the only content for c, c, c, etc.
                    # one coordinate ...
                    for comp in coord.split():
                        source[-1].append(float(comp)) # each component as data field / variable
                
    #print("target[0]", target[0])
    #print("source[0]", source[0])
    #print("target", target)
                
    return source, target

    '''
    for key,val in dictRaw.items():
        for key,val in dictRaw.items():
            print(key)
            break
    '''


def readFile(filename):
    with open(filename, "r") as f:
        content = f.read()
        return content


def writeFile(filename, content):
    #with open(self.getXMLPath(), "wb") as f:       # bytes
    with open(filename, "w") as f:
        f.write(content)
        print("writeFile(): file exported:", filename)


