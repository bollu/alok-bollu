/*
 * Isso vai ser sempre depois que a página carregar, é uma "main" só que não
 */
$(function() {

    //Previne o Submit do form ao apertar Enter
    $('form').bind("keypress", function(event) {
        if (event.keyCode == 13) {
            event.preventDefault();
            return false;
        }
    });

    //Quando pressionado Enter no campo de texto da suggestion, insere a nova option
    $('#inputWord').keypress(function(event) {
        if (event.keyCode == 13) {
            addSuggestion();
        }
    });

    //Quando pressionado "D" no campo da selects, deleta a opção selecionada
    $('#candidateList').keypress(function(event) {
        if (event.keyCode == 100) {
        	removeSelected();
        }
    });

});

function addSuggestion() {

    //busca a nova sugestão a ser inserida no text input
    var newSuggestion = $('#inputWord').val();
    if(newSuggestion.length != 0) { 
      appendNewOption(newSuggestion); 
    }
    //limpa o input
    $('#inputWord').val('');
}

function appendNewOption(textToNewOption) {
	
	//inicia uma diretriz do tipo option
    var newOption = document.createElement('option');

    //Adiciona o texto na nova option
    //Observação, sempre insira o VALUE da tua option, é isso que vai ser enviado
    newOption.text = textToNewOption;
    newOption.value = textToNewOption;

    //apenda a nova option
    $('#candidateList').append(newOption);
}

function removeSelected() {

	//seleciona uma lista de todas as opções selecionadas
    $('#candidateList :selected').remove();
}

function clearAll() {	
	//seleciona uma lista de todas as opções existente na select
    $('#candidateList option').remove();
}

function isValidForm(){
    var equiv = document.getElementById('candidateList');
	var nbEquiv = equiv.length;
	if (nbEquiv < 2) {
	  alert("Pour la question 3, vous devez fournir au moins 2 mots ou expressions équivalentes ou similaires!");
	  return false; // keep form from submitting
	}
	if(equiv[0].textContent.trim() == "" || equiv[1].textContent.trim() == "" ){
	  alert("Les suggestions d'équivalents ne peuvent pas être vides");
	  return false; // keep form from submitting
	}
	if( equiv[0].textContent.trim() == equiv[1].textContent.trim() ){
	  alert("Entrez au moins deux suggestions DIFFÉRENTES");
	  return false;
	}

	if( document.getElementById('questio11').checked==false &&
	    document.getElementById('questio12').checked==false &&
	    document.getElementById('questio13').checked==false &&
	    document.getElementById('questio14').checked==false &&
	    document.getElementById('questio15').checked==false &&
	    document.getElementById('questio16').checked==false){
      alert('La question 4 est obligatoire !');
      return false;
	}

	if( document.getElementById('questio21').checked==false &&
	    document.getElementById('questio22').checked==false &&
	    document.getElementById('questio23').checked==false &&
	    document.getElementById('questio24').checked==false &&
	    document.getElementById('questio25').checked==false &&
	    document.getElementById('questio26').checked==false){
      alert('La question 5 est obligatoire !');
      return false;
	}
	if( document.getElementById('questio31').checked==false &&
	    document.getElementById('questio32').checked==false &&
	    document.getElementById('questio33').checked==false &&
	    document.getElementById('questio34').checked==false &&
	    document.getElementById('questio35').checked==false &&
	    document.getElementById('questio36').checked==false){
      alert('La question 6 est obligatoire !');
      return false;
	}
	 // else form is good let it submit, of course you will 
	 // probably want to alert the user WHAT went wrong.

	 return true;
}

function selectAll() {
    var e = document.getElementById('candidateList');
    for(var i=0; i < e.options.length; i++)  {
	  e.getElementsByTagName('option')[i].selected = 'selected';	
    }
}

function checkValid() {
  if( isValidForm() ) {
    selectAll();
	var bn = document.getElementById('bttNext');
	bn.style.display='none';
	var dbn = document.createElement("button");
	dbn.appendChild(document.createTextNode("Attendez..."));
	dbn.setAttribute("class","btn btn-default");
	dbn.style.width = "100px";
	dbn.style.float="right";
	dbn.disabled = true;
	dbn.type="submit";
	bn.parentElement.appendChild(dbn);
	var bp = document.getElementById('bttPular');
	bp.style.display='none';	
	var dbp = document.createElement("button");
	dbp.appendChild(document.createTextNode("Attendez..."));
	dbp.setAttribute("class","btn btn-default");
	dbp.disabled = true;
	dbp.type="submit";
	bp.parentElement.appendChild(dbp);	
	//e.hide();
    //e.disabled=true; 
    //e.value='Enviando...';    
    return true;
  }
  return false;
}

function setValidoParaPular(){
	var e = document.getElementById('candidateList');
	var opt1 = document.createElement('option');
	var opt2 = document.createElement('option');
	opt1.appendChild(document.createTextNode("Dummy 1"));
	opt2.appendChild(document.createTextNode("Dummy 2"));
	opt1.style.display='none';
	opt2.style.display='none';
	e.innerHTML="";	
	e.appendChild(opt1);
	e.appendChild(opt2);
	document.getElementById('questio11').checked=true;
	document.getElementById('questio21').checked=true;
	document.getElementById('questio31').checked=true;
}
