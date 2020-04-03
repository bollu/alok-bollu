<html lang="en">
	<head>
		<meta http-equiv="content-type" content="text/html; charset=ISO-8859-1">
		<title>Interpretação de substantivos compostos</title>
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
        <h1>Interpretação de substantivos compostos</h1>
          <br/>
          	<?php          	
            if( isset($anno)){
                echo '<h3 style="color:red">O identificador '.$anno.' não foi encontrado.</h2>';
            }            
            if( isset($_POST['create_id'])) {
                if( strtolower($_POST['passphrase'])=="macacovelhomacaco"){ // HARDCODED PASSPHRASE!
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
                      echo '<h3><span style="color:red">Impossível criar identificador. Tem certeza que já não possui uma conta?</span></h3>';
                      unset($annotid);
                    }
                    else {
                      echo '<h3>Seu identificador é <span style="color:red">'.$annotid.'</span> - Anote-o, pois não receberá nenhum email de confirmação.</h2>';
                      $sql = "INSERT INTO " . $prefix . "users (anotador,name,surname,age,country) VALUES (\"" . $annotid . "\",\"" . $firstname . "\",\"" . $surname . "\",\"" . $age . "\",\"" . $country . "\");";
                      $result = mysql_query($sql, $conecta);
                    }
                    mysql_close($conecta);
                }
                else{
                    echo '<h3 style="color:red">O código de acesso está errado. <br/>Consulte o código no email de convite que recebeu.</h2>';
                }
            }
                
            
	        ?>

          <br/>
          <h4>Já tenho um identificador:</h4>
          <form class="col-lg-12" method="POST">
            <div class="input-group" style="width:340px;text-align:center;margin:0 auto;">
            <input class="form-control input-lg" title="Seus dados não serão disponibilizados a terceiros." placeholder="Informe seu identificador" type="text" name="annotator" <?php echo (isset($annotid)?'value="'.$annotid.'"':''); ?> >
              <span class="input-group-btn">
		<button class="btn btn-lg btn-primary" type="submit" name="ok" value="ok">Conectar</button></span>
            </div>
          </form>
        </div>
        
      </div> <!-- /row -->
  
</div>	

<div class="container">
	<br/>
	<br/>
	<center><h4>Ainda não tenho um identificador:</h4></center>	

<h2>1. Leia as instruções</h2>

<ul>
	<li>Estamos interessados em entender como as expressões são interpretadas por um falante nativo do português no dia-a-dia.</li>
	<li>Você vai ler uma expressão. Em seguida, irá avaliar qual a contribuição do sentido individual de cada palavra para o sentido da expressão como um todo.</li>
	<li> Para cada expressão, você lerá 3 frases em que a expressão aparece. Se você não entender a expressão, simplesmente pule para a próxima questão.</li>
	<li>Se a expressão tiver mais de um significado, considere <strong>somente</strong> aquele mostrado nas frases de exemplo.</li>
	<li>Algumas frases podem conter erros de digitação, uma vez que são da Internet. Se for esse o caso, basta ignorar esses erros de digitação.</li>
	<li>Para cada expressão você terá que sugerir equivalentes (sinônimos ou similares) que poderiam ser usados nas frases ao invés da expressão, mantendo o mesmo significado. </li>
	<li>Prefira sinônimos curtos, com 1 a 3 palavras sempre que possível. Se você digitar menos que 2 sinônimos, ou sinônimos sem sentido, seu trabalho será desconsiderado.</li>	
	<li>Em seguida, você vai responder 3 perguntas sobre o significado das palavras individuais da expressão. Para cada pergunta, basta clicar na opção que reflete o quanto você acha que as palavras individuais contribuem para o significado da expressão em uma escala de 0 (<em>Não, a palavra não contribui em nada para o significado</em>) a 5 (<em>Sim, a palavra contribui muito para o significado</em>). Os valores intermediários também devem ser utilizados para graduar o seu julgamento.</li>	
	<li>Não pense demais para responder as perguntas. Não existem respostas certas ou erradas (desde que você siga as instruções).</li>
	<li>Cada expressão só pode ser avaliada uma vez, você não poderá voltar atrás nas suas respostas uma vez que elas forem enviadas</li>
	<li>Este não é um teste de memória ou de inteligência. Nós realmente só queremos compreender melhor o uso de expressões em português.</li><!-- e como eles são interpretados por falantes nativos.</li>-->
	<li>Se você tiver algum problema, comentário, dúvida ou sugestão, fale conosco usando o campo opcional de comentários no final da página.</li>
</ul>

<!----------------------------------------------------------------------------->
<hl/>

<h2>2. Leia os exemplos</h2>

  <h2>CABEÇA DURA</h2>

<strong>Sentença : </strong> <em>João foi <u>cabeça dura</u> e não seguiu o conselho dos diretores da empresa.</em>
<br/>
<strong>Pergunta :</strong>Liste no mínimo 2 a 3 expressões equivalentes ou similares a <em>cabeça dura</em>
<br/>
<strong>Resposta Esperada : </strong>
<ul>
  <li><em>teimoso</em></li>
  <li><em>pessoa teimosa</em></li>
  <li><em>insistente</em></li>  
  <li><em>...</em></li>
</ul>
<br/>
<strong>Explicação</strong> : Se refere a indivíduo que é teimoso, que não aceita opiniões dos outros. Privilegie formulações curtas e cujo sentido é muito próximo de "cabeça dura". Evite frases longas e definições como "refere-se a uma pessoa que é teimosa ou insistente".


  <h2>INFERNO ASTRAL</h2>
  
<strong>Sentença :</strong> <em>  O João está vivendo o seu <u>inferno astral</u> e tudo que acontecer de negativo está relacionado a isso.</em>
<br/>
<strong>Pergunta : </strong><em>Inferno astral</em> é realmente/literalmente um <em>inferno</em>?
<br/>
<strong>Resposta Esperada : </strong> <img src="img/answer-1.png">
<br/>
<strong>Explicação</strong> : A expressão <em>inferno astral</em> se refere a um período negativo na vida de uma pessoa. Não é literalmente um inferno, e apenas tem em comum com essa palavra o fato de ser algo negativo.

<!----------------------------------------------------------------------------->
<hl/>

  <h2>CABRA-CEGA</h2>
<strong>Sentença :</strong> <em>As crianças da rua adoram brincar de <u>cabra-cega</u>.</em>
<br/>
<strong>Pergunta : </strong><em>Cabra-cega</em> é realmente/literalmente <em>cega</em>?
<br/>
<strong>Resposta Esperada : </strong> <img src="img/answer-0.png">
<br/>
<strong>Explicação</strong> : A expressão <em>cabra-cega</em> se refere a um jogo em que um dos participantes está de olhos vendados e tenta encontrar os outros. Não existe ninguém literalmente cego nesse jogo.


<!----------------------------------------------------------------------------->
<hl/>

  <h2>ARMÁRIO EMBUTIDO</h2>

<strong>Sentença : </strong> <em>Recomendamos a empresa para quem quer um <u>armário embutido</u> ideal em seu quarto. </em>
<br/>
<strong>Pergunta :</strong> Um <em>armário embutido</em> é realmente/literalmente um <em>armário</em> que está <em>embutido</em>?
<br/>
<strong>Resposta Esperada : </strong> <img src="img/answer-5.png">
<br/>
<strong>Explicação</strong> : A expressão <em>armário embutido</em> se refere a um armário construído sob medida junto das paredes de um ambiente. Ele é literalmente embutido na parede.


<!----------------------------------------------------------------------------->
<hl/>
          <h2>3. Preencha o formulário abaixo</h2>
      <div class="row">       
        <div class="col-lg-12 text-center v-center">
          <form class="col-lg-12" method="POST">
            <div class="input-group" style="width:380px;text-align:center;margin:0 auto;">
            <input required class="form-control input-lg" title="Digite aqui seu nome" placeholder="Nome (e.g. João)" type="text" name="name"/>
            <input required class="form-control input-lg" title="Digite aqui seu sobrenome" placeholder="Sobrenome (e.g. Silva)" type="text" name="surname"/>
            <input required class="form-control input-lg" title="Digite aqui sua idade" placeholder="Idade (e.g. 25)" type="text" name="age">
            <input required class="form-control input-lg" title="Onde você mora?" placeholder="País de residência (e.g. Brasil, Portugal...)" type="text" name="country">
            <input required class="form-control input-lg" title="Código de acesso" placeholder="Código recebido por email" type="text" name="passphrase">
            <p><input required class="" title="Se você não preenche esse critério, por favor não participe do nosso experimento" type="checkbox" name="native"/> Certifico que vivi no Brasil até os 13 anos de idade e que meus pais falaram português comigo durante esse período.</p>
		<button class="btn btn-lg btn-primary" type="submit" name="create_id" value="create_id">Criar meu identificador</button>
            </div>
          </form>
        </div>
        
      </div> <!-- /row -->
  
</div>

	</body>
</html>
