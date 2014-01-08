var mapper = function(){
    for(var i in this.headers.To){
	emit({From:this.headers.From, To:this.headers.To[i]}, 1)
    }

};

var reducer = function(correspondence, count){
    return Array.sum(count);
};

db.messages.mapReduce(mapper, reducer, {out: "map_reduce_out"})

db.map_reduce_out.find().sort({value: -1})
