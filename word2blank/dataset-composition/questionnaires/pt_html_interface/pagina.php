<?php
//header('Content-Type: text/html; charset=utf-8');
if(!isset($_COOKIE["annotator"])){
    header("location:index.php");
}
$prefix = "mturk_pt_";
$MAXANNOT = "30";
$GOAL=90;
include_once("db.php");

/******************************************************************************/

function get_random_mwe_id($anno){
    global $MAXANNOT;
    $conecta = get_db_connection();
    # Select MWE IDs that : are not annotated by current annotator and do not have MAXANNOT annotations yet
    $sql = "SELECT nan.id FROM (SELECT m.id, COUNT(m.id) as cid FROM (mturk_pt_mwes as m LEFT JOIN mturk_pt_respostas AS r ON m.id = r.idMWE) WHERE id NOT IN (SELECT idMWE FROM mturk_pt_respostas WHERE anotador = '" . $anno . "') GROUP BY m.id) AS nan WHERE cid <= " . $MAXANNOT . ";";    
    $result = mysql_query($sql, $conecta);
    $lstIds = array();
    if($result){
        while($consulta = mysql_fetch_array($result)) {            
            $lstIds[] = $consulta[0];        
            //echo "<li>" . $consulta[0] . "</li>";    
        }
    }
    mysql_close($conecta);
    $tent = $lstIds[array_rand($lstIds,1)];
    if(isset($tent)) {
        return $tent;
    }
    else{
        echo "<h1>Voc� anotou todas as express�es, obrigado! :-)</h1>";
    }    
}

/******************************************************************************/

function store_previous_answer($ans1, $ans2, $ans3, $comments, $equivalents, $anno){
    $idMWE = $_POST['idMWE'];
    $idSent= $_POST['idSent'];
    global $prefix;
    $conecta = get_db_connection();
    $escapedComments = mysql_escape_string($comments);
    $sql = "INSERT INTO " . $prefix . "respostas (idMWE, idSent, anotador, resp1, resp2, resp3, comments) VALUES (" . $idMWE . "," . $idSent . ",\"" . $anno . "\",".$ans1.",".$ans2.",".$ans3.",\"".$escapedComments."\"" . ")";
    $result = mysql_query($sql, $conecta);
    for($i=0; $i < count($equivalents); $i++){
        $escapedEquiv = mysql_escape_string($equivalents[$i]);
        $sql = "INSERT INTO " . $prefix . "anotacao (idmwe, idsent, idanno, word) VALUES (" . $idMWE . "," . $idSent . ",\"" . $anno . "\",\"" . $escapedEquiv . "\")";
        $result = mysql_query($sql, $conecta);
    }
    mysql_close($conecta);
}

/******************************************************************************/

//sleep(2); Test sending button disabled

$anno = $_COOKIE["annotator"];
// User skipped previous question, store this decision
if(isset($_POST['btt_pular'])){ 
    store_previous_answer(-1,-1,-1,"pulou",array(),$anno);   
}
// User submitted last question, store the answers
if(isset($_POST['btt_next'])){ 
    $equivalents = $_POST['values'];
    $ans1 = $_POST['Qhead'];
    $ans2 = $_POST['Qmodifier'];
    $ans3 = $_POST['Qheadmodifier'];
    $comments = $_POST['comments'];
    store_previous_answer($ans1, $ans2, $ans3, $comments, $equivalents,$anno);
}
// Generate next question
$idMWE = get_random_mwe_id($anno);
if($idMWE) { // else no new question available, all annotated or problem
    $conecta = get_db_connection();      
    //Retrieve all information about the compound
    $sql = "SELECT * FROM " . $prefix . "mwes where id = " . $idMWE; 
    $result = mysql_query($sql, $conecta);
    $mweinfo = mysql_fetch_assoc($result);
    // Retrieve information about user
    $sql = "SELECT count(*) FROM mturk_pt_respostas WHERE anotador = '". $anno . "';";
    $result = mysql_query($sql, $conecta);
    $done = mysql_fetch_row($result)[0];
    $percent = min(round((($done*100.0)/$GOAL)),100);
    mysql_close($conecta);

    $mweinfo['compound'] = $mweinfo['mwe'];

    echo '<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="pt">
  <head>
      <meta http-equiv="content-type" content="text/html; charset=ISO-8859-1">
      <title>Interpreta��o de substantivos compostos</title>
      <meta name="generator" content="Bootply" />
      <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
      <link href="css/bootstrap.min.css" rel="stylesheet">
      <!--[if lt IE 9]>
        <script src="//html5shim.googlecode.com/svn/trunk/html5.js"></script>
      <![endif]-->
      <link href="css/styles.css" rel="stylesheet">
      <link href="css/mturk.css" rel="stylesheet">        
  </head>
  <body>
  <div class="idandprogress">
    <p>Voc� est� conectado como <strong>: '.$anno.'</strong>, voc� j� anotou '.$done.' express�es, seu objetivo � '.$GOAL.' express�es (faltam '.max(0,$GOAL-$done).')</p>    
    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="'.$percent.'" aria-valuemin="0" aria-valuemax="100" style="width:'.$percent.'%">'.$percent.'%</div>            
    </div>
    <div class="panel panel-primary">      
    <div class="panel-heading"><strong>Instru��es (lembrete)</strong></div>
      <div class="panel-body">
        <p>Voc� ler� uma express�o da l�ngua portuguesa. Em seguida, avaliar� qual � a contribui��o de cada palavra individual para o sentido global da express�o.</p>
        <ul>
          <li>Cada express�o deve levar em torno de 1 minuto.</li>          
          <li>Pedimos que voc� leia as 3 frases de exemplo. Se voc� n�o compreend�-las, passe para a pr�xima quest�o.</li>
          <li>Se a express�o lhe parecer amb�gua, considere APENAS o sentido que aparecer nas frases de exemplo.</li>
          <li>Voc� s� avaliar� cada express�o uma vez, n�o � poss�vel revisar suas anota��es.</li>
          <li>N�o pense muito em cada pergunta, existem diversas respostas poss�veis.</li>
        </ul>
      </div>
    </div>
  </div>
  <form action="pagina.php" method="POST" onsubmit="return checkValid()">
    <INPUT TYPE="hidden" name="idMWE" VALUE="'. $idMWE . '">
    <INPUT TYPE="hidden" name="idSent" VALUE="'. $idMWE . '">
    <!-- script references -->
      <script src="js/jquery-2.1.1.min.js"></script>
      <script src="js/bootstrap.min.js"></script>
      <script src="js/suggestion_processing.js"></script>
      <div class="container-full">
        <div class="col-md-8">            
           
          <fieldset>
            <label>1. Leia a express�o abaixo:</label>
            <br/>
            <span class="indentation"></span><span style="font-size: 20pt"><em>' . $mweinfo['compound'] . '</em></span>
          </fieldset>
          
          <br/><!--=====================================================-->
              
          <fieldset>
            <label>2. Leia as seguintes frases contendo a express�o <em>' . $mweinfo['compound'] . '</em>:</label>
            <br/>
            <ul>
              <li>' . $mweinfo["examplesent1"] . '</li>
              <li>' . $mweinfo["examplesent2"] . '</li>
              <li>' . $mweinfo["examplesent3"] . '</li>
            </ul>
            <hr/>
            <em> N�o entendi o significado dessa express�o nas frases &#8594; </em> 
            <button onclick="setValidoParaPular()" class="btn btn-default" type="submit" name="btt_pular" value="skipPage" id="bttPular"> Pular essa express�o </button>            
          </fieldset>            
          
          <br/><!--=====================================================-->
          
          <fieldset>
            <label>3. Forne�a 2 a 3 palavras ou express�es sin�nimas ou similares a <em>' . $mweinfo['compound'] . '</em>:</label>
            <br/>
            <!--<div style="width:400px;text-align:center;margin:0 auto;"><font color="black">Use ENTER para adicionar a resposta. Para deletar a resposta selecionada use DELETE</font></div>-->
            <br/>
            <div class="input-group" style="width:400px;text-align:center;margin:0 auto;">
                <input id="inputWord" class="form-control input-lg" title="Prefira sugest�es curtas, de 1 a 3 palavras, se poss�vel utilizando as palavras &quot;' . $mweinfo['noun'] . '&quot; e/ou &quot;' . $mweinfo['modifier'] . '&quot;" placeholder="Equivalentes de '. $mweinfo['compound'] . '..." type="text">
                <span class="input-group-btn"><button id="submitWord" onclick="addSuggestion()" class="btn btn-lg btn-primary" type="button">enter</button></span>
            </div>
            <br/>
            <select id="candidateList" class="form-control" multiple="multiple" style="width:400px;margin:0 auto;" name="values[ ]"></select>
            <br/>
            <center>
                <button onclick="removeSelected()" class="btn btn-default" style="width:200px;" type="button">Apagar</button>
                <button onclick="clearAll()" class="btn btn-default" style="width:200px;" type="button">Apagar tudo</button>
            </center>
          </fieldset>
          
          <br/><!--=====================================================-->                
                    
          <fieldset>
            <label>4. Na sua vis�o, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' sempre literalmente ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em>? </label>
            <br/>
            <br/>            
            <table class="radio-table">
              <tbody>
                <tr><td class="bigno" rowspan="2">N�O</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">SIM</td></tr>
                <tr>
                  <td class="tooltippy"><input id="questio11"  type="radio" name="Qhead" value="0"/><div class="ttip">De forma nenhuma &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> n�o ' . $mweinfo['ter'] . ' <u>nada a ver</u> com ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio12"  type="radio" name="Qhead" value="1"/><div class="ttip">N�o &mdash; eu vejo apenas uma <u>rela��o vaga</u> entre ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> e ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio13"  type="radio" name="Qhead" value="2"/><div class="ttip">N�o exatamente &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>associado</u> ao de <em>' . $mweinfo['noun'] . '</em>, mas apenas <u>indiretamente</u></div></td>
                  <td class="tooltippy"><input id="questio14"  type="radio" name="Qhead" value="3"/><div class="ttip">De certa forma &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>diretamente associado</u> ao de <em>' . $mweinfo['noun'] . '</em>, mesmo que n�o sejam sentidos id�nticos</div></td>
                  <td class="tooltippy"><input id="questio15"  type="radio" name="Qhead" value="4"/><div class="ttip">Sim &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> <u>' . $mweinfo['ser'] . ' mesmo</u> ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em>, em um sentido pouco comum da palavra <em>' . $mweinfo['noun'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio16"  type="radio" name="Qhead" value="5"/><div class="ttip">Com certeza &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' <u>sempre literalmente</u> ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em></div></td>
                </tr>
              </tbody>
            </table>
          </fieldset>
          
          <br/><!--=====================================================-->
          
          <fieldset>
            <label>5. Na sua vis�o, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' sempre literalmente <em>' . $mweinfo['modifier'] . '</em>? </label>
            <br/>
            <br/>            
            <table class="radio-table">
              <tbody>
                <tr><td class="bigno" rowspan="2">N�O</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">SIM</td></tr>
                <tr>
                  <td class="tooltippy"><input id="questio21"  type="radio" name="Qmodifier" value="0"/><div class="ttip">De forma nenhuma &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> n�o ' . $mweinfo['ter'] . ' <u>nada</u> de <em>' . $mweinfo['modifierLemma'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio22"  type="radio" name="Qmodifier" value="1"/><div class="ttip">N�o &mdash; eu vejo apenas uma <u>rela��o vaga</u> entre ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> e algo <em>' . $mweinfo['modifierLemma'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio23"  type="radio" name="Qmodifier" value="2"/><div class="ttip">N�o exatamente &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>associado</u> ao de algo <em>' . $mweinfo['modifierLemma'] . '</em>, mas apenas <u>indiretamente</u></div></td>
                  <td class="tooltippy"><input id="questio24"  type="radio" name="Qmodifier" value="3"/><div class="ttip">De certa forma &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>diretamente associado</u> ao de algo <em>' . $mweinfo['modifierLemma'] . '</em>, mesmo que n�o sejam sentidos id�nticos</div></td>
                  <td class="tooltippy"><input id="questio25"  type="radio" name="Qmodifier" value="4"/><div class="ttip">Sim &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> <u>' . $mweinfo['ser'] . ' mesmo</u> <em>' . $mweinfo['modifier'] . '</em>, em um sentido pouco comum da palavra <em>' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio26"  type="radio" name="Qmodifier" value="5"/><div class="ttip">Com certeza &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' <u>sempre literalmente</u> <em>' . $mweinfo['modifier'] . '</em></div></td>
                </tr>
              </tbody>
            </table>
          </fieldset>
          
          <br/><!--=====================================================-->
          <fieldset>
            <label>6. Dadas essas respostas, pode-se dizer que ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' sempre literalmente ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> que ' . $mweinfo['ser'] . ' <em>' . $mweinfo['modifier'] . '</em>? </label>
            <br/>
            <br/>          
            <table class="radio-table">
              <tbody>
                <tr><td class="bigno" rowspan="2">N�O</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">SIM</td></tr>
                <tr>
                  <td class="tooltippy"><input id="questio31"  type="radio" name="Qheadmodifier" value="0"/><div class="ttip">De forma nenhuma &mdash; <u>n�o faz sentido</u> imaginar ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> que ' . $mweinfo['ser'] . ' ' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio32"  type="radio" name="Qheadmodifier" value="1"/><div class="ttip">N�o &mdash; � <u>estranho</u> imaginar ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> que ' . $mweinfo['ser'] . ' <em>' . $mweinfo['modifier'] . '</em>, mesmo que o sentido seja compreens�vel</div></td>
                  <td class="tooltippy"><input id="questio33"  type="radio" name="Qheadmodifier" value="2"/><div class="ttip">N�o exatamente &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>associado</u> a ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> e a algo <em>' . $mweinfo['modifierLemma'] . '</em>, mas essa associa��o n�o � direta</div></td>
                  <td class="tooltippy"><input id="questio34"  type="radio" name="Qheadmodifier" value="3"/><div class="ttip">De certa forma &mdash; o sentido de <em>' . $mweinfo['compound'] . '</em> est� <u>diretamente associado</u> a ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> e algo <em>' . $mweinfo['modifierLemma'] . '</em>, mesmo que n�o sejam sentidos id�nticos</div></td>
                  <td class="tooltippy"><input id="questio35"  type="radio" name="Qheadmodifier" value="4"/><div class="ttip">Sim &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> <u>' . $mweinfo['ser'] . ' mesmo</u> ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> que ' . $mweinfo['ser'] . ' <em>' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio36"  type="radio" name="Qheadmodifier" value="5"/><div class="ttip">Com certeza &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['ser'] . ' <u>sempre literalmente</u> ' . $mweinfo['undefdetHead'] . ' <em>' . $mweinfo['noun'] . '</em> que ' . $mweinfo['ser'] . ' <em>' . $mweinfo['modifier'] . '</em></div></td>
                </tr>
              </tbody>
            </table>
          </fieldset>
          
          <br/><!--=====================================================-->
    
          <fieldset>
            <label>Voc� pode usar o campo abaixo caso tenha coment�rios ou sugest�es sobre esta quest�o. </label><br/>
            <textarea cols="40" rows="5" name="comments"></textarea>
          </fieldset>
          
          <br/><!--=====================================================-->    
          
          <button class="btn btn-default" style="width:100px;float: right; margin-bottom:20px;" type="submit" name="btt_next" value="nextPage" id="bttNext">Enviar</button>
        </div>            
      </div>
    </form>
  </body>
</html>';
}
?>
