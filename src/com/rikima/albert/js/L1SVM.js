function L1SVM(wv, bias, c, exampleN) {
    this.wv = wv;
    this.bias = bias;
    this.exampleN = exampleN;
    this.round = 0;
    this.lambda = c;
}

L1SVM.prototype.get_eta = function() {
    return 1.0 / (1.0 + this.round / this.exampleN);
}

L1SVM.prototype.score = function(fv) {
    return this.wv.dot(fv) + this.bias;
}


L1SVM.prototype.classify = function(fv) {
    var s = this.score(fv);
    return (s > 0)? 1 : -1;
}

L1SVM.prototype.learn = function(fv) {
    if (fv.y * this.score(fv) < 1.0) {
	var eta = this.get_eta(this.round);
	for (var f in fv.features) {
	    var v = fv.features[f];
	    var s = v * fv.y * eta;
	    if (this.wv.features[f] == undefined) {
		this.wv.features[f] = 0;	
	    }
	    this.wv.features[f] += s;
	}
    }
}

L1SVM.prototype.l1regularize = function() {
    var lambda_hat = this.get_eta() * this.lambda;
    for (var f in this.wv.features) {
	var v = this.wv.features[f];	    
	var abs_v = Math.abs(v);
	var new_v = 0.0;
	var diff = (abs_v - lambda_hat);

	if (diff > 0.0) {
	    new_v = diff;
	    if (v < 0.0) {
		new_v *= -1;
	    }
	}
	this.wv.features[f] = new_v;
    }
}
