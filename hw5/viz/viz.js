d3.csv("date_counts.csv", function(error, data){

    var tweets = data.map(function(d){return +d.Tweets})
    var MAX_POINT = Math.max.apply(Math, tweets)

    var NUM_POINTS = data.length;
    var BAR_WIDTH = 20;
    var BAR_SPACING = 10;
    var MAX_BAR_HEIGHT = 500;

    var svg = d3.select('#graph')
        .append('svg')
        .attr('width', BAR_WIDTH * BAR_SPACING * NUM_POINTS )
        .attr('height', MAX_BAR_HEIGHT);

    svg.selectAll(".bar")
	.data(data)
	.enter()
	.append("rect")
	.attr("height",function(d){
	    return (+d.Tweets)/MAX_POINT*MAX_BAR_HEIGHT
	})
        .attr("width",BAR_WIDTH)
        .attr("x",function(d,i){ 
	    return i*(BAR_WIDTH + BAR_SPACING) 
	})
        .attr("y",function(d){
	    return MAX_BAR_HEIGHT - (+d.Tweets)/MAX_POINT*MAX_BAR_HEIGHT
	})
        .style("fill","red");
    
});
