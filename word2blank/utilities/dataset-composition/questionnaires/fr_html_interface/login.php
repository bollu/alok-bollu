<html lang="en">
	<head>
		<meta http-equiv="content-type" content="text/html; charset=ISO-8859-1">
		<title>Interpr�tation de noms compos�s</title>
		<meta name="generator" content="Bootply" />
		<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
		<link href="css/bootstrap.min.css" rel="stylesheet">
		<link href="//netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.min.css" rel="stylesheet">
		<!--[if lt IE 9]>
			<script src="//html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<![endif]-->
		<link href="css/styles.css" rel="stylesheet">
	</head>
	<body>
	
<div class="container">
      <div class="row">       
        <div class="col-lg-12 text-center v-center"> 
        <h1>Interpr�tation de noms compos�s</h1>
          <br/>
          	<?php          	
            if( isset($anno)){
                echo '<h3 style="color:red">L\'identifiant '.$anno.' n\'a pas �t� retrouv�.</h2>';
            }            
            if( isset($_POST['create_id'])) {
                if( strtolower($_POST['passphrase'])=="carteblanchecartebleue"){ // HARDCODED PASSPHRASE!
                    include_once("db.php");                
                    $conecta =  get_db_connection();                    
                    $firstname = mysql_escape_string($_POST['name']);
                    $surname = mysql_escape_string($_POST['surname']);
                    $age = mysql_escape_string($_POST['age']);
                    $country = mysql_escape_string($_POST['country']); 
                    $notnew = true; 
                    $tries = 0;                  
                    while( $notnew and $tries <= 10 ) {
                      $prefixid = strtolower(str_replace(' ', '-', $surname . $firstname . "user" )); // Replaces all spaces with hyphens.
                      $prefixid = preg_replace('/[^a-z0-9\-]/', '', $prefixid); // Remove special chars
                      $annotid =  substr ( $prefixid , 0, 8) . rand( 100000 , 999999 );
                      $sql = "SELECT anotador FROM " . $prefix . "users WHERE anotador = '" . $annotid . "'";
                      $result = mysql_query($sql, $conecta);
                      $notnew = mysql_fetch_row($result);
                      $tries++;                   
                    }
                    if( $tries >= 10 ){
                      echo '<h3><span style="color:red">Imposs�vel criar identificador. Tem certeza que j� n�o possui uma conta?</span></h3>';
                      unset($annotid);
                    }
                    else {
                      echo '<h3>Votre identifiant est <span style="color:red">'.$annotid.'</span> - Notez-le car vous <strong>n\'allez pas</strong> recevoir un mail de confirmation.</h2>';
                      $sql = "INSERT INTO " . $prefix . "users (anotador,name,surname,age,country) VALUES (\"" . $annotid . "\",\"" . $firstname . "\",\"" . $surname . "\",\"" . $age . "\",\"" . $country . "\");";
                      $result = mysql_query($sql, $conecta);
                    }
                    mysql_close($conecta);
                }
                else{
                    echo '<h3 style="color:red">Le code anti-spam est faux. <br/>Entrez le code anti-spam qui vous a �t� communiqu� dans un mail par les responsables de l\'exp�rience.</h2>';
                }
            }
                
            
	        ?>

          <br/>
          <h4>J'ai d�j� mon identifiant:</h4>
          <form class="col-lg-12" method="POST">
            <div class="input-group" style="width:340px;text-align:center;margin:0 auto;">
            <input class="form-control input-lg" title="Vos informations personnelles ne seront pas diffus�es." placeholder="Votre identifiant" type="text" name="annotator" <?php echo (isset($annotid)?'value="'.$annotid.'"':''); ?> >
              <span class="input-group-btn">
		<button class="btn btn-lg btn-primary" type="submit" name="ok" value="ok">Login</button></span>
            </div>
          </form>
        </div>
        
      </div> <!-- /row -->
  
</div>	

<div class="container">
	<br/>
	<br/>
	<center><h4>Je n'ai pas encore d'identifiant:</h4></center>	

<h2>1. Instructions</h2>

<ul>
  <!--<li>Notre objectif est de mieux comprendre comment les noms compos�s sont interpr�t�s en fran�ais par des locuteurs natifs dans le langage courant.</li>
  <li>Vous allez lire un nom compos� en fran�ais. Ensuite, vous allez �valuer quelle est la contribution du sens individuel de chaque mot au sens global du nom compos�.</li>
  <li>Nous vous demandons de lire 3 phrases dans lesquelles le nom compos� appara�t. Si vous ne les comprenez pas, passez � l'expression suivante.</li>
  <li>Si ce nom compos� vous semble ambig�, consid�rez <strong>uniquement</strong> le sens qui appara�t dans les phrases.</li>
  <li>Certaines de ces phrases peuvent contenir des coquilles, car elles ont �t� trouv�es sur Internet. Vous pouvez ignorer ces coquilles.</li>
   <li>Pour chaque nom compos�, proposez des formulations alternatives avec un sens �quivalent ou proche.</li>     
   <li>Privil�giez les formulations courtes, avec 1 � 3 mots si possible. Vous devez donner au moins 2 propositions diff�rentes pour que votre contribution ne soit pas �cart�e.</li> 
  <li>Ensuite, vous r�pondrez � 3 questions concernant les mots qui composent le nom compos�. S�lectionnez une valeur entre 0 et 5, o� 0 signifie <em>pas du tout</em> et 5 signifie <em>tout � fait</em>. Vous devez utiliser les valeurs interm�diaires pour nuancer votre r�ponse.</li>  
  <li>Ne r�fl�chissez pas trop pour chaque question, il n'y a pas de mauvaise r�ponse.</li>
  	<li>Chaque mot compos� sera propos� une seule fois, vous ne pouvez pas modifier votre r�ponse apr�s l'avoir envoy�e</li>
  <li>Ceci n'est pas un test d'intelligence ou de m�moire. Notre but est uniquement de comprendre comment ces mots compos�s sont utilis�s en fran�ais.</li>
  <li>Si vous avez des probl�mes, remarques ou suggestions, n'h�sitez pas � nous en faire en remplissant le champs optionnel des commentaires, en bas de chaque page.</li> -->
  
  <li>Notre objectif est de comprendre comment les noms compos�s sont interpr�t�s en fran�ais. </li> 
  <li>Vous allez lire un nom compos�, puis �valuer quelle est la contribution de chaque mot au sens du nom compos�.</li>
  <li>D'abord, lisez les 3 phrases contenant ce nom compos�. Si vous ne les comprenez pas, passez la question.</li>
   <li>Ensuite, proposez des formulations alternatives avec un sens �quivalent ou proche. Il n'est pas n�cessaire de donner des synonymes parfaits :-)</li>     
  <li>Puis, vous r�pondrez � 3 questions concernant les mots qui composent le nom compos�. S�lectionnez une valeur entre 0 et 5, o� 0 signifie <em>pas du tout</em> et 5 signifie <em>tout � fait</em>. Utilisez les valeurs interm�diaires pour nuancer votre r�ponse.</li>  
  <li>Ne r�fl�chissez pas trop pour chaque question, il n'y a pas de mauvaise r�ponse.</li>
</ul>

<!----------------------------------------------------------------------------->
<hl/>

<h2>2. Exemples</h2>

<!----------------------------------------------------------------------------->


  <h2>CARTON PLEIN</h2>

<strong>Phrase : </strong> <em>Si l'Espagne parvient � faire <u>carton plein</u> face au Br�sil, elle sera en bonne position pour atteindre la finale.</em>
<br/>
<strong>Question :</strong> Donnez des alternatives pour dire <em>carton plein</em>
<br/>
<strong>R�ponse attendue : </strong>
<ul>
  <li><em>victoire</em></li>
  <li><em>r�ussite</em></li>
  <li><em>grand succ�s</em></li>  
</ul>
<br/>
<strong>Explication</strong> : faire carton plein c'est r�ussir, avoir du succ�s. Les alternatives propos�es ont un sens proche tout en restant assez concises.

<!----------------------------------------------------------------------------->

  <h2>BOUC �MISSAIRE</h2>
  
<strong>Phrase :</strong> <em>J�r�me Kerviel est le <u>bouc �missaire</u> d'un syst�me bancaire opaque et tentaculaire</em>
<br/>
<strong>Question :</strong> Un <em>bouc �missaire</em> est-il vraiment/litt�ralement un <em>bouc</em> ?
<br/>
<strong>R�ponse attendue : </strong> <img src="img/answer-0.png">
<br/>
<strong>Explication</strong> : dans la phrase, il s'agit bien d'une personne. Elle ne se transforme pas litt�ralement en bouc, c'est un sens figur�. 


<!----------------------------------------------------------------------------->


  <h2>NUIT BLANCHE</h2>

<strong>Phrase : </strong> <em>Si vous faites une <u>nuit blanche</u>, vous serez tr�s fatigu� et in�vitablement vous ferez beaucoup d'erreurs lors de l'examen.</em>
<br/>
<strong>Question :</strong> Une <em>nuit blanche</em> est-elle vraiment/litt�ralement <em>blanche</em> ?
<br/>
<strong>R�ponse attendue : </strong> <img src="img/answer-1.png">
<br/>
<strong>Explication</strong> : la nuit n'a pas vraiment de couleur, mais on pourrait dire qu'elle est associ�e au noir. Si on reste r�veill� toute la nuit, elle ne change pas de couleur, donc le lien est vague.


<!----------------------------------------------------------------------------->


  <h2>COMPTE BANCAIRE</h2>

<strong>Phrase : </strong> <em>Les cl�tures abusives de <u>compte bancaire</u> sont rares, car toute banque peut fermer votre compte, d�s lors qu'elle respecte un pr�avis.</em>
<br/>
<strong>Question :</strong> Un <em>compte bancaire</em> est-it vraiment/litt�ralement un <em>compte</em> qui est <em>bancaire</em> ?
<br/>
<strong>R�ponse attendue : </strong> <img src="img/answer-5.png">
<br/>
<strong>Explication</strong> : un compte bancaire est un compte, ni plus ni moins. Il s'agit bien d'un compte en banque, donc bancaire. Les deux mots sont employ�s litt�ralement.



<!----------------------------------------------------------------------------->
<hl/>
          <h2>3. Formulaire d'Inscription</h2>
      <div class="row">       
        <div class="col-lg-12 text-center v-center">
          <form class="col-lg-12" method="POST">
            <div class="input-group" style="width:380px;text-align:center;margin:0 auto;">
            <input required class="form-control input-lg" title="Entrez ici votre nom" placeholder="Nom (p. ex. Dupont)" type="text" name="surname"/>
            <input required class="form-control input-lg" title="Entrez ici votre pr�nom" placeholder="Pr�nom (p. ex. Jean)" type="text" name="name"/>
            <input required class="form-control input-lg" title="Entrez ici votre �ge" placeholder="�ge (p. ex. &quot;25&quot;)" type="text" name="age">
            <input required class="form-control input-lg" title="Dans quel pays habitez-vous ?" placeholder="Pays de r�sidence (p. ex. France, Belgique, Canada ...)" type="text" name="country">
            <input required class="form-control input-lg" title="Code anti-spam" placeholder="Code anti-spam re�u par mail" type="text" name="passphrase">
            <p><input required class="" title="Si vous ne remplissez pas ce crit�re, nous vous prions de ne pas participer � cette enqu�te" type="checkbox" name="native"/> Je certifie que j'ai v�cu en France pendant mon enfance jusqu'� mes 13 ans, et que mes parents ont parl� fran�ais avec moi pendant cette p�riode.</p>
		<button class="btn btn-lg btn-primary" type="submit" name="create_id" value="create_id">Cr�er mon identifiant</button>
            </div>
          </form>
        </div>
        
      </div> <!-- /row -->
  
</div>

	</body>
</html>
