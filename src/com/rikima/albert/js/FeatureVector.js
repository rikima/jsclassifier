function FeatureVector(y) {
    this.y = y;
    this.features = {};
}

FeatureVector.prototype.dim = function() {
    var c = 0;
    for (var f in this.features) {
	if (Math.abs(this.features[f]) > 0) {
	    c += 1;
	}
    }
    return c;    
}

FeatureVector.prototype.dot = function(fv) {
    var hs_s;
    var hs_l;
    if (this.features.length < fv.length) {
	hs_s = this.features;
	hs_l = fv.features;
    }
    else {
	hs_s = fv.features;
	hs_l = this.features;
    }

    var v = 0;
    for (var f in hs_s) {
	if (hs_l[f] != undefined) {
	    v += hs_l[f] * hs_s[f];
	}
    }
    return v;
}

FeatureVector.prototype.toString = function() {
    var rv = 'y=' + this.y + ' ';
    for (var f in this.features) {
	rv += f + ':' + this.features[f] + ' '
    }
    return rv;
}