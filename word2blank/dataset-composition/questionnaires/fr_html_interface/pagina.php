<?php
//header('Content-Type: text/html; charset=utf-8');
if(!isset($_COOKIE["annotator"])){
    header("location:index.php");
}
$prefix = "mturk_fr_";
$MAXANNOT = "50";
$GOAL=20;
include_once("db.php");

/******************************************************************************/

function get_random_mwe_id($anno){
    global $MAXANNOT;
    global $prefix;
    $conecta = get_db_connection();
    # Select MWE IDs that : are not annotated by current annotator and do not have MAXANNOT annotations yet
    $sql = "SELECT nan.id FROM (SELECT m.id, COUNT(m.id) as cid FROM (" . $prefix . "mwes as m LEFT JOIN " . $prefix . "respostas AS r ON m.id = r.idMWE) WHERE id NOT IN (SELECT idMWE FROM " . $prefix . "respostas WHERE anotador = '" . $anno . "') GROUP BY m.id) AS nan WHERE cid <= " . $MAXANNOT . ";";
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
        echo "<h1>Vous avez annot� toutes les expressions, merci ! :-)</h1>";
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
    $sql = "SELECT count(*) FROM " . $prefix . "respostas WHERE anotador = '". $anno . "';";
    $result = mysql_query($sql, $conecta);
    $done = mysql_fetch_row($result)[0];
    $percent = min(round((($done*100.0)/$GOAL)),100);
    mysql_close($conecta);

    $mweinfo['compound'] = $mweinfo['mwe'];

    echo '<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="pt">
  <head>
      <meta http-equiv="content-type" content="text/html; charset=ISO-8859-1">
      <title>Interpr�tation de noms compos�s</title>
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
    <p>Votre identifiant est <strong>: '.$anno.'</strong>, vous avez d�j� �valu� '.$done.' expressions, votre objectif est d\'annoter '.$GOAL.' expressions (il en reste '.max(0,$GOAL-$done).')</p>    
    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="'.$percent.'" aria-valuemin="0" aria-valuemax="100" style="width:'.$percent.'%">'.$percent.'%</div>            
    </div>
    <div class="panel panel-primary">      
    <div class="panel-heading"><strong>Instructions (rappel)</strong></div>
      <div class="panel-body">
        <p>Vous allez lire un nom compos� en fran�ais. Ensuite, vous allez �valuer quelle est la contribution du sens de chaque mot au sens du nom compos�.</p>
        <ul>
          <li>Lisez 3 phrases contenant le nom compos�. Si vous ne les comprenez pas, passez la question.</li>
          <li>Si ce nom compos� vous semble ambigu, consid�rez UNIQUEMENT le sens qui appara�t dans les 3 phrases. </li>
          <li>Vous jugerez chaque expression une seule fois, vous ne pouvez pas revenir en arri�re.</li>
          <li>Ne r�fl�chissez pas trop pour chaque question, plusieurs r�ponses correctes sont possibles.</li>
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
            <label>1. Lisez l\'expression ci-dessous :</label>
            <br/>
            <span class="indentation"></span><span style="font-size: 20pt"><em>' . $mweinfo['compound'] . '</em></span>
          </fieldset>
          
          <br/><!--=====================================================-->
              
          <fieldset>
            <label>2. Lisez les phrases ci-dessous, contenant l\'expression <em>' . $mweinfo['compound'] . '</em>:</label>
            <br/>
            <ul>
              <li>' . $mweinfo["examplesent1"] . '</li>
              <li>' . $mweinfo["examplesent2"] . '</li>
              <li>' . $mweinfo["examplesent3"] . '</li>
            </ul>
            <hr/>
            <em> Je ne comprends pas le sens de cette expression dans ces phrases &#8594; </em> 
            <button onclick="setValidoParaPular()" class="btn btn-default" type="submit" name="btt_pular" value="skipPage" id="bttPular"> Expression suivante </button>            
          </fieldset>            
          
          <br/><!--=====================================================-->
          
          <fieldset>
            <label>3. Donnez 2 � 3 mots ou expressions �quivalentes <u>ou proches</u> � <em>' . $mweinfo['compound'] . '</em>:</label>
            <br/>
            <!--<div style="width:400px;text-align:center;margin:0 auto;"><font color="black">Use ENTER para adicionar a resposta. Para deletar a resposta selecionada use DELETE</font></div>-->
            <br/>
            <div class="input-group" style="width:400px;text-align:center;margin:0 auto;">
                <input id="inputWord" class="form-control input-lg" title="Pr�f�rez les formulations courtes, de 1 � 3 mots, en utilisant les mots &quot;' . $mweinfo['noun'] . '&quot; et/ou &quot;' . $mweinfo['modifier'] . '&quot; quand c\'est possible. Les propositions ne doivent pas forc�ment �tre des synonymes." placeholder="'. $mweinfo['compound'] . ' est similaire � ..." type="text">
                <span class="input-group-btn"><button id="submitWord" onclick="addSuggestion()" class="btn btn-lg btn-primary" type="button">Entr�e</button></span>
            </div>
            <br/>
            <select id="candidateList" class="form-control" multiple="multiple" style="width:400px;margin:0 auto;" name="values[ ]"></select>
            <br/>
            <center>
                <button onclick="removeSelected()" class="btn btn-default" style="width:200px;" type="button">Supprimer</button>
                <button onclick="clearAll()" class="btn btn-default" style="width:200px;" type="button">Tout supprimer</button>
            </center>
          </fieldset>
          
          <br/><!--=====================================================-->                
                    
                    <fieldset>
            <label>4. � votre avis, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' forc�ment/litt�ralement ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> ? </label>
            <br/>
            <br/>
            <table class="radio-table">
              <tbody>
              <tr>
                  <td class="bigno" rowspan="2">NON</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">OUI</td>
              </tr>
              <tr>
                  <td class="tooltippy"><input id="questio11" type="radio" name="Qhead" value="0"/><div class="ttip">Pas du tout &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> n\'' . $mweinfo['avoir'] . ' <u>rien � voir</u> avec ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio12" type="radio" name="Qhead" value="1"/><div class="ttip">Non &mdash; je peux imaginer un <u>lien tr�s vague</u> entre ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> et ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em><!---, mais c\'est tir� par les cheveux--></div></td>
                  <td class="tooltippy"><input id="questio13" type="radio" name="Qhead" value="2"/><div class="ttip">Plut�t non &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>li�</u> � ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em>, mais <u>pas directement</u></div></td>
                  <td class="tooltippy"><input id="questio14" type="radio" name="Qhead" value="3"/><div class="ttip">Plut�t oui &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>directement li�</u> � ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em>, m�me s\'ils ne sont pas �quivalents</div></td>
                  <td class="tooltippy"><input id="questio15" type="radio" name="Qhead" value="4"/><div class="ttip">Ouais &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ', <u>parfois</u>, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em>, dans un sens peu usuel du mot <em>' . $mweinfo['noun'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio16" type="radio" name="Qhead" value="5"/><div class="ttip">Tout � fait &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' <u>forc�ment/litt�ralement</u> ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em></div></td>
              </tr>
              </tbody>
            </table>
          </fieldset>

          <br/><!--=====================================================-->

          <fieldset>
            <label>5. � votre avis, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' forc�ment/litt�ralement <em>' . $mweinfo['modifier'] . '</em> ? </label>
            <br/>
            <br/>
            <table class="radio-table">
              <tbody>
              <tr>
                  <td class="bigno" rowspan="2">NON</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">OUI</td>
              </tr>
              <tr>
                  <td class="tooltippy"><input id="questio21"  type="radio" name="Qmodifier" value="0"/><div class="ttip">Pas du tout &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> n&apos;' . $mweinfo['avoir'] . ' <u>rien � voir</u> avec quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio22"  type="radio" name="Qmodifier" value="1"/><div class="ttip">Non &mdash; je peux imaginer un <u>lien tr�s vague</u> entre ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> et quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio23"  type="radio" name="Qmodifier" value="2"/><div class="ttip">Plut�t non &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>li�</u> � quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em>, mais <u>pas directement</u></div></td>
                  <td class="tooltippy"><input id="questio24"  type="radio" name="Qmodifier" value="3"/><div class="ttip">Plut�t oui &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>directement li�</u> � quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em>, m�me s&apos;ils ne sont pas �quivalents</div></td>
                  <td class="tooltippy"><input id="questio25"   type="radio" name="Qmodifier" value="4"/><div class="ttip">Ouais &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ', <u>parfois</u>, <em>' . $mweinfo['modifier'] . '</em>, dans un sens peu usuel du mot <em>' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio26"  type="radio" name="Qmodifier" value="5"/><div class="ttip">Tout � fait &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' <u>forc�ment/litt�ralement</u> <em>' . $mweinfo['modifier'] . '</em></div></td>
              </tr>
              </tbody>
            </table>
          </fieldset>

          <br/><!--=====================================================-->

          <fieldset>
            <label>6. �tant donn� ces r�ponses, peut-on dire ' . $mweinfo['queDet'] . '' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' forc�ment/litt�ralement ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> qui ' . $mweinfo['etre'] . ' <em>' . $mweinfo['modifier'] . '</em> ? </label>
            <br/>
            <br/>
            <table class="radio-table">
              <tbody>
              <tr>
                  <td class="bigno" rowspan="2">NON</td> <td class="number">0</td> <td class="number">1</td> <td class="number">2</td> <td class="number">3</td> <td class="number">4</td> <td class="number">5</td> <td class="bigyes" rowspan="2">OUI</td>
              </tr>
              <tr>
                  <td class="tooltippy"><input id="questio31" type="radio" name="Qheadmodifier" value="0"/><div class="ttip">Pas du tout &mdash; cela n\'a <u>pas de sens</u> d\'imaginer ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> qui ' . $mweinfo['etre'] . ' <em>' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio32" type="radio" name="Qheadmodifier" value="1"/><div class="ttip">Non &mdash; c\'est <u>bizarre</u> d\'imaginer ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> qui ' . $mweinfo['etre'] . ' <em>' . $mweinfo['modifier'] . '</em>, m�me si �a reste compr�hensible</div></td>
                  <td class="tooltippy"><input id="questio33" type="radio" name="Qheadmodifier" value="2"/><div class="ttip">Plut�t non &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>li�</u> � ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> et � quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em>, mais le lien n\'est pas direct</div></td>
                  <td class="tooltippy"><input id="questio34" type="radio" name="Qheadmodifier" value="3"/><div class="ttip">Plut�t oui &mdash; le sens de <em>' . $mweinfo['compound'] . '</em> est <u>directement li�</u> � ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> et � quelque chose de <em>' . $mweinfo['modifierLemma'] . '</em>, m�me s\'ils ne sont pas �quivalents</div></td>
                  <td class="tooltippy"><input id="questio35" type="radio" name="Qheadmodifier" value="4"/><div class="ttip">Ouais &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ', <u>parfois</u>, ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> qui ' . $mweinfo['etre'] . ' <em>' . $mweinfo['modifier'] . '</em></div></td>
                  <td class="tooltippy"><input id="questio36" type="radio" name="Qheadmodifier" value="5"/><div class="ttip">Tout � fait &mdash; ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['compound'] . '</em> ' . $mweinfo['etre'] . ' <u>forc�ment/litt�ralement</u> ' . $mweinfo['undefdet'] . ' <em>' . $mweinfo['noun'] . '</em> qui ' . $mweinfo['etre'] . ' <em>' . $mweinfo['modifier'] . '</em></div></td>
              </tr>
              </tbody>
            </table>
          </fieldset>
          
          <br/><!--=====================================================-->            
    
          <fieldset>
            <label>7. Entrez ici des commentaires libres sur cette t�che : </label><br/>
            <span class="tooltippy"><span class="indentation"></span><textarea cols="40" rows="5" name="comments"></textarea><span class="ttip">Remplissez ce champ uniquement si vous avez des probl�mes, des questions ou des suggestions</span></span>
          </fieldset>
          
          <br/><!--=====================================================-->    
          
          <button class="btn btn-default" style="width:100px;float: right; margin-bottom:20px;" type="submit" name="btt_next" value="nextPage" id="bttNext">Envoyer</button>
        </div>            
      </div>
    </form>
  </body>
</html>';
}
?>
