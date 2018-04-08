var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var Mouse = {x:0, y:0};
var Mouse_prev = {x:0, y:0};

var clearButton = document.getElementById("clear");
var submitButton = document.getElementById("submit");
var painterSize = document.getElementById("painterSize");

$('#title').hide().fadeIn(1000);
$('#result').hide();

var DEBUG = false;

ctx.lineWidth = painterSize.value;

function draw(){
		
		ctx.lineJoin = "round";

		ctx.beginPath();
		ctx.moveTo(Mouse_prev.x, Mouse_prev.y);
		ctx.lineTo(Mouse.x, Mouse.y);
		ctx.closePath();
		ctx.stroke();

	}

function start(){
	this.addEventListener('mousemove', draw, false);
}

function stop(){
	this.removeEventListener("mousemove", draw, false);
}

function move(e){

	Mouse_prev.x = Mouse.x;
	Mouse_prev.y = Mouse.y;

    Mouse.x = e.pageX - this.offsetLeft;
    Mouse.y = e.pageY - this.offsetTop;

    if (DEBUG){
    	var mouse = document.getElementById("mousePosition");
    	mouse.textContent = "mouse at: " + String(Mouse.x) + ", " + String(Mouse.y);
	}

}

function clear(){
	ctx.clearRect(0, 0, canvas.width, canvas.height);
}


function submit(){
	// var image = canvas.toDataURL("image.png").replace("image.png", "octet-stream");
	// window.location.href=image;

	var image = canvas.toDataURL();

//	$.ajax({
//		type: "POST",
//		url:  "/mix/",
//		data: {
//			imageBase64: image
//		},
//		success: function(){
//			console.log("submit succeeded.");
//		}
//	});
	$.post("/mix/", {imageBase64: image}).done(function(data){
	    console.log("submit succeeded.");
	    console.log(data);
	    submitButton.disabled = false;
	    submitButton.value = "submit";
	    $('#result').show();
	    $('#result').attr("src", "/static/result.jpg");
	    console.log($('#result').attr("src"));
	})

	submitButton.disabled = true;
	submitButton.value = "processing";

}

function resize(){
    ctx.lineWidth = painterSize.value;
}


canvas.addEventListener('mousedown', start, false);
canvas.addEventListener('mouseup', stop, false);
canvas.addEventListener('mousemove', move, false);
clearButton.addEventListener('click', clear, false);
submitButton.addEventListener('click', submit, false);
painterSize.addEventListener('change', resize, false);
