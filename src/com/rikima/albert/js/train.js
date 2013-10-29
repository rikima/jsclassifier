importPackage(java.io);

load('FeatureVector.js');
load('L1SVM.js');

br=BufferedReader( InputStreamReader( FileInputStream("/home/rikitoku/workspace//jsclassifier/data/a9a.svmdata") ) );


fvs = []
while( line= br.readLine() ){
    var fv = new FeatureVector(1);
    if( line==null ){
        break ;
    }

    var ss = line.split(' ');
    var y = parseInt(ss[0]);
    fv.y = y;
    for (var i = 1; i < ss.length; i++) {
	var s = ss[i]
	var ss2 = s.split(':')
	if (ss2.length == 2) {
	    var f = ss2[0];
	    var v = parseInt(ss2[1]);

	    if (fv.features[f] == undefined) {
		fv.features[f] = 0;	
	    }
	    
	    fv.features[f] += v;
	}
    }
    fvs.push(fv);
}
br.close();
print('#fvs', fvs.length);

var wv = new FeatureVector(1);
var bias = 0.0;
var c = 0.01;
var exampleN = fvs.length;

var learner = new L1SVM(wv, bias, c, exampleN);

var maxRound = 20;
for (var r = 0; r < maxRound; ++r) {
    for (var i = 0; i < fvs.length;++i) {
	fv = fvs[i];
        try {
	    learner.learn(fv);
	} catch (x) {
	    print(x)
	    print('fv', fv)
	}
    }
    learner.l1regularize(r);
    learner.round += 1;
}

var pp = 0;
var pn = 0;
var np = 0;
var nn = 0;
for (var i = 0; i < fvs.length;++i) {
    fv = fvs[i];

    try {
	var s = learner.score(fv);
	print('y:' + fv.y + ' score:' + s);
	
	if (fv.y > 0) {
	    if (s > 0) {
		pp += 1;
	    }
	    else {
		pn += 1;
	    }
	}
	else {
	    if (s > 0) {
		np += 1;
	    }
	    else {
		nn += 1;
	    }
	}
    } catch (x) {
	print(x)
	print('fv', fv)
    }
} 
print('pp', pp, 'pn', pn, 'np', np, 'nn', nn);
print('#wv', wv.dim());